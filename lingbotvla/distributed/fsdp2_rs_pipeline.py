# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Relax the FSDP2 backward ReduceScatter depth-1 lock (monkey-patch, torch 2.8.0).

torch 2.8.0 hard-codes RS pipeline depth 1: `_fsdp_param_group.py:429-436` makes
the *compute* stream wait on the previous group's RS completion event before the
next group's copy-in, so the reduce-scatter input buffer can be recycled. There
is no public knob in 2.8.0 (`set_custom_reduce_scatter` is 2.9+).

Mechanism, buffer-lifetime proof, memory cost and A/B plan:
report/_artifacts/rs_pipeline_depth/NOTES.md
"""

import importlib
import os
from collections import deque
from typing import Any, Optional

from ..utils import logging


logger = logging.get_logger(__name__)

_DEPTH_ENV = "LINGBOT_FSDP2_RS_DEPTH"
_MAX_DEPTH = 8
_RING_ATTR = "_lingbot_rs_ring"
_MARK_ATTR = "_lingbot_rs_patch"

# Module search order: torch>=2.7 keeps FSDP2 internals under
# torch.distributed.fsdp._fully_shard; 2.4-2.6 kept them under
# torch.distributed._composable.fsdp (which is a re-export shim in 2.8.0).
_MODULE_CANDIDATES = (
    ("torch.distributed.fsdp._fully_shard._fsdp_param_group",
     "torch.distributed.fsdp._fully_shard._fsdp_state"),
    ("torch.distributed._composable.fsdp._fsdp_param_group",
     "torch.distributed._composable.fsdp._fsdp_state"),
)


def _rs_depth() -> int:
    """In-flight reduce-scatters allowed. 1 = do not patch torch at all."""
    raw = os.environ.get(_DEPTH_ENV, "1")
    try:
        depth = int(raw)
    except ValueError:
        logger.warning_rank0(f"{_DEPTH_ENV}={raw!r} is not an int; falling back to 1.")
        return 1
    if depth > _MAX_DEPTH:
        logger.warning_rank0(f"{_DEPTH_ENV}={depth} clamped to {_MAX_DEPTH}.")
        return _MAX_DEPTH
    return max(1, depth)


def _resolve_classes():
    for pg_name, st_name in _MODULE_CANDIDATES:
        try:
            pg_mod = importlib.import_module(pg_name)
            st_mod = importlib.import_module(st_name)
            return pg_mod.FSDPParamGroup, st_mod.FSDPState
        except Exception:
            continue
    return None, None


def _ring(comm_ctx: Any) -> deque:
    ring = getattr(comm_ctx, _RING_ATTR, None)
    if ring is None:
        ring = deque()
        setattr(comm_ctx, _RING_ATTR, ring)
    return ring


def _retire(device_handle: Any, state: Optional[Any]) -> None:
    """Order the compute stream past one RS, then let its input buffer go.

    Same invariant as upstream `_fsdp_param_group.py:429-436`: the RS input is
    allocated on the compute stream (`_fsdp_collectives.py:437`, outside the
    `with device_handle.stream(reduce_scatter_stream)` block that starts at
    `:452`) and read by NCCL on the RS stream (`:461`) with no `record_stream()`
    anywhere in FSDP2, so the block may only return to the caching allocator's
    compute-stream pool after the compute stream has been ordered past the RS
    completion event recorded at `:467`.
    """
    if state is not None and state.event is not None:
        device_handle.current_stream().wait_event(state.event)


def maybe_patch_reduce_scatter_pipeline() -> int:
    """Patch FSDP2 for depth-N backward RS if `LINGBOT_FSDP2_RS_DEPTH` > 1.

    Returns the effective depth. Depth 1 (default) returns without importing or
    mutating anything in torch, so the control arm is bit-identical.
    """
    depth = _rs_depth()
    if depth <= 1:
        return 1

    param_group_cls, state_cls = _resolve_classes()
    if param_group_cls is None or state_cls is None:
        logger.warning_rank0(
            f"{_DEPTH_ENV}={depth}: FSDP2 internals not found under any known "
            "module path; RS pipeline patch NOT applied (depth stays 1)."
        )
        return 1
    if getattr(param_group_cls.post_backward, _MARK_ATTR, False):
        return depth

    orig_post_backward = param_group_cls.post_backward
    orig_final_callback = state_cls._root_post_backward_final_callback

    def post_backward(self, *unused: Any) -> None:
        comm_ctx = self.comm_ctx
        ring = _ring(comm_ctx)
        # Emptying `reduce_scatter_state` is what disables the lock: upstream
        # :429-436 then finds `None`, skips its wait, and its clear is a no-op.
        if comm_ctx.reduce_scatter_state is not None:
            ring.append(comm_ctx.reduce_scatter_state)
            comm_ctx.reduce_scatter_state = None
        while len(ring) >= depth:
            _retire(self.device_handle, ring.popleft())
        orig_post_backward(self, *unused)
        if comm_ctx.reduce_scatter_state is not None:
            ring.append(comm_ctx.reduce_scatter_state)
            comm_ctx.reduce_scatter_state = None

    def _root_post_backward_final_callback(self) -> None:
        # Must run after `orig`: the loop at _fsdp_state.py:299-305 still calls
        # post_backward for groups autograd skipped, so the ring can grow there.
        orig_final_callback(self)
        if self._state_ctx.is_last_backward:
            ring = _ring(self._comm_ctx)
            while ring:
                _retire(self._device_handle, ring.popleft())

    setattr(post_backward, _MARK_ATTR, True)
    setattr(post_backward, "__wrapped__", orig_post_backward)
    setattr(_root_post_backward_final_callback, _MARK_ATTR, True)
    setattr(_root_post_backward_final_callback, "__wrapped__", orig_final_callback)
    param_group_cls.post_backward = post_backward
    state_cls._root_post_backward_final_callback = _root_post_backward_final_callback

    logger.info_rank0(
        f"{_DEPTH_ENV}={depth}: FSDP2 backward ReduceScatter depth-1 lock relaxed "
        f"to depth={depth} (<= {depth - 1} RS left un-waited while the next "
        f"copy-in runs); patched {param_group_cls.__module__}.FSDPParamGroup."
        "post_backward + FSDPState._root_post_backward_final_callback."
    )
    return depth
