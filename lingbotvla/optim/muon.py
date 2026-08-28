"""DTensor-aware Muon optimizer for FSDP2 and MoE expert weights.

``DistributedMuon`` keeps upstream ``torch.optim.Muon`` numerics for 2D
weights and adds batched Newton-Schulz for 3D MoE expert stacks.

For FSDP2-sharded 2D params, same-shape parameters are mega-batched:
stacked into a single tensor, gathered with one NCCL call, orthogonalized
with one batched NS pass, and scattered back locally.
"""

from collections import defaultdict
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.optim.optimizer import Optimizer


try:
    from torch.optim._muon import (
        DEFAULT_A,
        DEFAULT_B,
        DEFAULT_C,
        DEFAULT_NS_STEPS,
        EPS,
        _adjust_lr,
    )

    _MUON_AVAILABLE = True
except ImportError:  # pragma: no cover - torch < 2.9 fallback
    _MUON_AVAILABLE = False
    DEFAULT_A = 3.4445
    DEFAULT_B = -4.7750
    DEFAULT_C = 2.0315
    DEFAULT_NS_STEPS = 5
    EPS = 1e-7

    def _adjust_lr(  # type: ignore[no-redef]
        lr: float,
        adjust_lr_fn: Optional[str],
        param_shape: Sequence[int],
    ) -> float:
        """Torch 2.8 fallback for ``torch.optim._muon._adjust_lr``."""
        if adjust_lr_fn is None:
            return lr
        fan_out, fan_in = param_shape[:2]
        if adjust_lr_fn == "original":
            return lr * math.sqrt(max(1.0, fan_out / fan_in))
        if adjust_lr_fn == "match_rms_adamw":
            return lr * 0.2 * math.sqrt(max(fan_out, fan_in))
        raise ValueError(f"Adjust learning rate function {adjust_lr_fn} is not supported")


__all__ = [
    "DEFAULT_NS_COEFFICIENTS",
    "DEFAULT_NS_STEPS",
    "DistributedMuon",
    "batched_newton_schulz",
    "split_muon_adamw_params",
]


DEFAULT_NS_COEFFICIENTS: Tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C)

_DEFAULT_ADAMW_NAME_PATTERNS: Tuple[str, ...] = (
    "embed_tokens",
    "embedding",
    "lm_head",
    "output_layer",
)

_MEGABATCH_MAX_GROUP_SIZE = 32

# AG/NS pipelining. Rationale + measured payoff table:
# memory/experiments/plan-muon-p1a-260818.md
_PIPELINE_DEPTH_ENV = "LINGBOT_MUON_PIPELINE_DEPTH"
_CHUNK_ORDER_ENV = "LINGBOT_MUON_CHUNK_ORDER"
_AG_BYTE_CAP_ENV = "LINGBOT_MUON_AG_BYTE_CAP"
_AG_DTYPE_ENV = "LINGBOT_MUON_AG_DTYPE"

# Newton-Schulz sharded to chunk owners. 0 = off (every rank runs every NS).
# Rounds, mechanism and the measured p2p bandwidth law:
# report/_docs/muon_ns_shard_a2a_proposal.md, memory/.../exp-muon-ns-a2a-*.md
_NS_SHARD_ENV = "LINGBOT_MUON_NS_SHARD"

# Newton-Schulz runs in this dtype no matter what it is handed; the AG downcast
# below is only safe because it matches. Keep the two tied.
NS_COMPUTE_DTYPE = torch.bfloat16


def _pipeline_depth() -> int:
    """AG chunks allowed in flight. 1 = legacy fully-serial path."""
    try:
        return max(1, int(os.environ.get(_PIPELINE_DEPTH_ENV, "2")))
    except ValueError:
        return 2


def _chunk_order() -> str:
    """``bytes_desc`` (default) or ``legacy`` (pre-260818 shape-key order)."""
    return os.environ.get(_CHUNK_ORDER_ENV, "bytes_desc").strip().lower()


def _ag_byte_cap() -> int:
    """On-wire AG bytes allowed per chunk. 0 = count-based chunking only."""
    try:
        return max(0, int(os.environ.get(_AG_BYTE_CAP_ENV, "0")))
    except ValueError:
        return 0


def _ag_dtype() -> Optional[torch.dtype]:
    """Wire dtype for the Muon all-gather. ``None`` = keep the param dtype.

    ``bf16`` is provably bitwise-identical: NS casts to bf16 on its first line,
    and cast-then-cat == cat-then-cast (both elementwise). See
    memory/experiments/exp-muon-ag-bf16-260819.md.
    """
    name = os.environ.get(_AG_DTYPE_ENV, "").strip().lower()
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16}.get(name)


def _ns_shard_rounds() -> int:
    """Number of owner-exchange rounds per step. 0 = replicated NS (legacy)."""
    try:
        return max(0, int(os.environ.get(_NS_SHARD_ENV, "0")))
    except ValueError:
        return 0


def _ns_cost(shape: tuple, count: int) -> float:
    """Relative NS cost of a chunk, from global metadata only.

    NS transposes to M <= K, then does 3 matmuls per step on [B, M, K]:
    flops ~ 2*B*M^2*(2K + M).  Bytes are 2*B*M*K, so cost per byte scales with
    the SHORT edge -- measured spread across production shapes is 5.4x, which is
    why ``ag_bytes`` (a comm ordering) is the wrong key for balancing NS.
    """
    rows, cols = (shape + (1, 1))[:2] if len(shape) < 2 else shape[-2:]
    m, k = min(rows, cols), max(rows, cols)
    return float(count) * m * m * (2 * k + m)


@torch.no_grad()
def batched_newton_schulz(
    grad: Tensor,
    ns_coefficients: Tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
    ns_steps: int = DEFAULT_NS_STEPS,
    eps: float = EPS,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Run quintic Newton-Schulz on each trailing ``[M, K]`` matrix."""
    if ns_steps >= 100:
        raise ValueError("Number of steps must be less than 100 for computational efficiency")
    if grad.ndim < 2:
        raise ValueError(f"Input must have ndim >= 2, got shape {tuple(grad.shape)}")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")

    a, b, c = ns_coefficients
    original_dtype = grad.dtype
    ortho = grad.to(compute_dtype)

    transposed = ortho.size(-2) > ortho.size(-1)
    if transposed:
        ortho = ortho.mT

    norm = ortho.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    ortho = ortho / norm

    for _ in range(ns_steps):
        A = ortho @ ortho.mT
        if A.ndim == 2:
            gram_update = torch.addmm(A, A, A, beta=b, alpha=c)
            ortho = torch.addmm(ortho, gram_update, ortho, beta=a)
        else:
            *batch, M_, K_ = ortho.shape
            B_ = 1
            for d in batch:
                B_ *= d
            A_3d = A.reshape(B_, M_, M_)
            ortho_3d = ortho.reshape(B_, M_, K_)
            gram_update = torch.baddbmm(A_3d, A_3d, A_3d, beta=b, alpha=c)
            ortho = torch.baddbmm(ortho_3d, gram_update, ortho_3d, beta=a).reshape(*batch, M_, K_)
        del A

    if transposed:
        ortho = ortho.mT

    return ortho.to(original_dtype)


def _is_adamw_by_name(name: str, extra_patterns: Sequence[str]) -> bool:
    lname = name.lower()
    for pat in _DEFAULT_ADAMW_NAME_PATTERNS:
        if pat in lname:
            return True
    for pat in extra_patterns:
        if pat and pat.lower() in lname:
            return True
    return False


def _is_muon_eligible_ndim(param: Tensor) -> bool:
    """Return True for dense linears and 3D MoE expert stacks."""
    return param.ndim in (2, 3)


def split_muon_adamw_params(
    model: "nn.Module",
    no_decay_modules: Optional[List[str]] = None,
    no_decay_params: Optional[List[str]] = None,
    extra_adamw_name_patterns: Optional[Sequence[str]] = None,
) -> Tuple[List[Tensor], List[Tensor], List[str], List[str]]:
    """Split model parameters into Muon-eligible weights and AdamW fallback weights."""
    no_decay_modules = no_decay_modules or []
    no_decay_params = no_decay_params or []
    extra_patterns = list(extra_adamw_name_patterns or ())

    forced_adamw_fqns: set = set()
    for module_name, module in model.named_modules():
        cls_name = module.__class__.__name__
        is_embedding = isinstance(module, nn.Embedding)
        is_no_decay = cls_name in no_decay_modules
        if is_embedding or is_no_decay:
            for pname, _p in module.named_parameters(recurse=False):
                fqn = f"{module_name}.{pname}" if module_name else pname
                forced_adamw_fqns.add(fqn)

    muon_params: List[Tensor] = []
    adamw_params: List[Tensor] = []
    muon_names: List[str] = []
    adamw_names: List[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        muon_ok = _is_muon_eligible_ndim(param)
        forced_adamw = (
            (not muon_ok)
            or name in forced_adamw_fqns
            or _is_adamw_by_name(name, extra_patterns)
            or any(p and p.lower() in name.lower() for p in no_decay_params)
        )
        if forced_adamw:
            adamw_params.append(param)
            adamw_names.append(name)
        else:
            muon_params.append(param)
            muon_names.append(name)

    return muon_params, adamw_params, muon_names, adamw_names


_KIND_LOCAL = "local"
_KIND_FSDP_GATHER_2D = "fsdp_gather_2d"
_KIND_MOE_LOCAL_3D = "moe_local_3d"
_KIND_MOE_GATHER_3D = "moe_gather_3d"


# --------------------------------------------------------------------------- #
# Instrumentation: per-chunk NVTX + metadata for the DistributedMuon AG/NS
# pipeline (see memory/experiments/plan-muon-p1a-260818.md).
#
# 6 NVTX ranges per chunk:
#   muon_group/{i}   ..  whole chunk (depth=1 only; depth>1 interleaves chunks)
#     muon_pack/{i}  ..  Phase 1 momentum + Phase 2 stack/pad
#     muon_ag/{i}    ..  all_gather ISSUE (returns immediately when async)
#     muon_wait/{i}  ..  work.wait() + reconstruction cat/narrow == exposed AG
#     muon_ns/{i}    ..  batched Newton-Schulz on the gathered tensor
#     muon_apply/{i} ..  narrow-view scatter + per-param add_ into params
# The END of `muon_apply/{i}` is the last read of chunk i's AG-derived buffer.
#
# Chunk numbering (`{i}`) is monotonic across the whole `.step()` call and
# identical on every rank, so multi-rank nsys traces align by chunk_idx.
# --------------------------------------------------------------------------- #

_MUON_PROFILE_ENV = "LINGBOT_MUON_PROFILE"


def _muon_profile_enabled() -> bool:
    """Whether to emit per-chunk NVTX ranges and collect chunk metadata.

    Enable with ``LINGBOT_MUON_PROFILE=1`` in the training env. Off by
    default so unrelated runs pay zero cost.
    """
    return os.environ.get(_MUON_PROFILE_ENV, "").lower() in ("1", "true", "yes", "on")


class _MuonPerfRange:
    """CUDA NVTX + torch.profiler.record_function; no-op when ``enabled=False``.

    Mirrors the ``_perf_range`` helper in ``tasks/vla/train_lingbotvla.py``
    (kept local here so the optimizer stays free of task-side imports).
    Every rank emits when enabled — required for multi-rank nsys alignment.
    """

    __slots__ = ("_name", "_enabled", "_rf")

    def __init__(self, name: str, enabled: bool) -> None:
        self._name = name
        self._enabled = enabled
        self._rf = None

    def __enter__(self):
        if self._enabled:
            torch.cuda.nvtx.range_push(self._name)
            self._rf = torch.profiler.record_function(self._name)
            self._rf.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._enabled and self._rf is not None:
            self._rf.__exit__(exc_type, exc_val, exc_tb)
            torch.cuda.nvtx.range_pop()
        return False


def _shard_dims(p: DTensor) -> List[int]:
    """Return the list of tensor dims along which ``p`` is sharded."""
    return [pl.dim for pl in p.placements if isinstance(pl, Shard)]


def _classify_param(p: Tensor) -> str:
    """Return one of ``_KIND_*`` describing how Muon should treat ``p``."""
    if not isinstance(p, DTensor):
        return _KIND_LOCAL

    shard_dims = _shard_dims(p)
    if not shard_dims:
        return _KIND_LOCAL

    if p.ndim == 2:
        return _KIND_FSDP_GATHER_2D

    if p.ndim == 3:
        if all(d == 0 for d in shard_dims):
            return _KIND_MOE_LOCAL_3D
        return _KIND_MOE_GATHER_3D

    raise ValueError(
        f"DistributedMuon got an unexpected param rank {p.ndim} "
        f"(shape={tuple(p.shape)}). Only 2D and 3D params are supported."
    )


def _full_grad(grad: Tensor) -> Tensor:
    """Return a replicated tensor, all-gathering DTensor gradients if needed."""
    if isinstance(grad, DTensor):
        return grad.full_tensor()
    return grad


def _wrap_full_as_dtensor_like(full: Tensor, ref: Tensor) -> Tensor:
    """Wrap ``full`` as a DTensor with ``ref``'s placements."""
    if not isinstance(ref, DTensor):
        return full

    mesh = ref.device_mesh
    replicated = DTensor.from_local(
        full,
        device_mesh=mesh,
        placements=[Replicate()] * mesh.ndim,
        run_check=False,
    )
    return replicated.redistribute(device_mesh=mesh, placements=ref.placements)


def _get_dtensor_shard_info(p: DTensor) -> Tuple[Any, int, int, int]:
    """Extract (process_group, world_size, rank, shard_dim) from a sharded DTensor."""
    mesh = p.device_mesh
    for mesh_dim_idx, placement in enumerate(p.placements):
        if isinstance(placement, Shard):
            pg = mesh.get_group(mesh_dim_idx)
            ws = mesh.size(mesh_dim_idx)
            rk = mesh.get_local_rank(mesh_dim_idx)
            return pg, ws, rk, placement.dim
    raise ValueError(f"No Shard placement found in DTensor with placements={p.placements}")


class DistributedMuon(Optimizer):
    """Muon optimizer with mega-batched Newton-Schulz for FSDP2-sharded params.

    Same-shape FSDP2-sharded 2D parameters are processed together: one batched
    all-gather, one batched NS, then local scatter. This reduces NCCL calls from
    O(num_params) to O(num_unique_shapes).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: Tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: Optional[str] = None,
        enable_nvtx: bool = False,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= float(lr):
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= float(momentum):
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= float(weight_decay):
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in ("original", "match_rms_adamw"):
            raise ValueError(f"Adjust learning rate function {adjust_lr_fn} is not supported")

        defaults: Dict[str, Any] = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)

        # Instrumentation state — always initialized; controls whether NVTX
        # ranges and per-chunk metadata are emitted. Effective enable = ctor
        # kwarg OR env var LINGBOT_MUON_PROFILE=1.
        self._enable_nvtx: bool = bool(enable_nvtx) or _muon_profile_enabled()
        # Chunk-level metadata list, appended once per _megabatch_finish
        # call. Reset at the start of every step() so `.step_index` is
        # monotonic within a step and step-to-step comparisons are trivial.
        # Entries collect (in insertion / call order):
        #   {
        #     "chunk_idx":         monotonic int within this step,
        #     "shape":             tuple(global param shape),
        #     "dtype":             str(param dtype),
        #     "param_count":       N params in this chunk (grad and no-grad),
        #     "with_grad_count":   how many had a real grad,
        #     "world_size":        pg world size,
        #     "shard_dim":         which dim is FSDP-sharded,
        #     "local_size":        local shard size on this rank pre-pad,
        #     "max_local_size":    ceil(global/world) — the actual AG chunk size,
        #     "ag_in_bytes":       nbytes of the tensor handed to all_gather,
        #     "ag_out_bytes":      cumulative nbytes of gather_list buffers,
        #     "pack_ms":           Phase 1+early-Phase 2 CUDA elapsed_time,
        #     "ag_ms":             AG issue -> AG complete (spans other chunks'
        #                          compute when pipelined; NOT the exposed cost),
        #     "wait_ms":           work.wait() -> AG complete == EXPOSED AG,
        #     "ns_ms":             Phase 3 batched NS elapsed_time,
        #     "apply_ms":          Phase 4 scatter+apply elapsed_time,
        #     "pipeline_depth":    AG chunks allowed in flight (1 = serial),
        #     "last_read_iter":    Phase 4 last iter index that touched buffer,
        #   }
        # ``_muon_events`` holds the raw (start, end) event pairs; timings are
        # computed lazily via ``dump_stats(...)`` after CUDA sync at step end.
        self._muon_stats: List[Dict[str, Any]] = []
        self._muon_events: List[Dict[str, Any]] = []
        self._muon_step_counter: int = 0
        self._muon_chunk_counter: int = 0

    def set_enable_nvtx(self, enabled: bool) -> None:
        """Toggle instrumentation at runtime (e.g. during profile window only)."""
        self._enable_nvtx = bool(enabled) or _muon_profile_enabled()

    def _make_event_pair(self) -> Tuple["torch.cuda.Event", "torch.cuda.Event"]:
        """Return a (start, end) CUDA event pair with timing enabled."""
        return (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    def dump_stats(self, path: str, rank: int = 0) -> None:
        """Sync CUDA, resolve event timings, and dump per-chunk metadata to JSON.

        Only call this from the rank whose stats you want to persist (typically
        rank 0). ``self._muon_stats`` and ``self._muon_events`` are cleared
        after a successful dump so subsequent steps start fresh.
        """
        import json  # local import — dump_stats is a rare / offline path

        if not self._muon_events:
            return

        # Force in-order completion of all recorded ranges before elapsed_time.
        torch.cuda.synchronize()

        for stat, ev in zip(self._muon_stats, self._muon_events):
            stat["pack_ms"] = ev["pack_s"].elapsed_time(ev["pack_e"])
            stat["ag_ms"] = ev["ag_s"].elapsed_time(ev["ag_e"])
            stat["ns_ms"] = ev["ns_s"].elapsed_time(ev["ns_e"])
            stat["apply_ms"] = ev["apply_s"].elapsed_time(ev["apply_e"])
            if "wait_s" in ev:
                # Exposed AG time. Σwait_ms is the number the pipeline must
                # shrink; Σag_ms cannot, since it now spans other chunks' NS.
                stat["wait_ms"] = ev["wait_s"].elapsed_time(ev["wait_e"])

        payload = {
            "rank": int(rank),
            "step_index": int(self._muon_step_counter),
            "num_chunks": len(self._muon_stats),
            "chunks": self._muon_stats,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=False)

        # Clear so the caller can call dump_stats every N steps without leaks.
        self._muon_stats = []
        self._muon_events = []

        for group in self.param_groups:
            for p in group["params"]:
                if not _is_muon_eligible_ndim(p):
                    raise ValueError(
                        "DistributedMuon supports only 2D and 3D parameters; "
                        f"got param with shape {tuple(p.size())}. Route 1D/4D+ "
                        "params (biases, norms, conv weights) to AdamW via "
                        "split_muon_adamw_params."
                    )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Reset per-step counters. `_muon_stats` and `_muon_events` accumulate
        # across steps until dump_stats() clears them — this lets the caller
        # dump once at the end of the profile window instead of every step.
        self._muon_step_counter += 1
        self._muon_chunk_counter = 0

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_coefficients = tuple(group["ns_coefficients"])
            ns_steps = int(group["ns_steps"])
            eps = float(group["eps"])
            adjust_lr_fn = group["adjust_lr_fn"]

            group_config = {
                "lr": lr,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "nesterov": nesterov,
                "ns_coefficients": ns_coefficients,
                "ns_steps": ns_steps,
                "eps": eps,
                "adjust_lr_fn": adjust_lr_fn,
            }

            # Classify ALL params upfront. Group FSDP_GATHER_2D by global shape.
            # CRITICAL: every rank must issue the same collective calls in the same
            # order, so we include ALL params (even grad=None) in the grouping and
            # skip the actual update for grad=None params inside the batch.
            fsdp_2d_groups: Dict[tuple, List[Tensor]] = defaultdict(list)
            other_params: List[Tuple[Tensor, str]] = []

            for p in group["params"]:
                kind = _classify_param(p)
                if kind == _KIND_FSDP_GATHER_2D:
                    global_shape = tuple(p.shape)  # DTensor .shape = global, same on all ranks
                    key = (global_shape, str(p.dtype))
                    fsdp_2d_groups[key].append(p)
                else:
                    if p.grad is None:
                        continue
                    if torch.is_complex(p):
                        raise RuntimeError("DistributedMuon does not support complex parameters")
                    if p.grad.is_sparse:
                        raise RuntimeError("DistributedMuon does not support sparse gradients")
                    other_params.append((p, kind))

            # --- Mega-batch path for FSDP_GATHER_2D ---
            # Flat chunk plan across all shape groups (largest AG first), run
            # through a depth-N AG/NS pipeline. Order is derived from global
            # shape/dtype/count only => identical collective order on all ranks.
            plan = self._build_megabatch_plan(fsdp_2d_groups, group_config)
            self._run_megabatch_pipeline(plan, group_config)

            # --- Per-param fallback for remaining kinds ---
            # Operate on local tensors to avoid DTensor dispatch issues
            # (torch.compile may produce non-DTensor grads that conflict with
            # DTensor in-place ops).
            for p, kind in other_params:
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                buf = state["momentum_buffer"]

                buf_local = buf.to_local() if isinstance(buf, DTensor) else buf
                grad_local = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad

                buf_local.lerp_(grad_local, 1 - momentum)
                if nesterov:
                    update_local = grad_local.lerp(buf_local, momentum)
                else:
                    update_local = buf_local.clone()

                # Compute ortho on local tensors
                if kind == _KIND_LOCAL or kind == _KIND_MOE_LOCAL_3D:
                    ortho_local = batched_newton_schulz(update_local, ns_coefficients, ns_steps, eps)
                elif kind == _KIND_MOE_GATHER_3D:
                    assert isinstance(p, DTensor)
                    # Need full tensor for NS — gather, compute, slice back
                    full_update = _full_grad(
                        DTensor.from_local(update_local, device_mesh=p.device_mesh, placements=p.placements, run_check=False)
                    )
                    full_ortho = batched_newton_schulz(full_update, ns_coefficients, ns_steps, eps)
                    # Take this rank's local shard back
                    pg, ws, rk, sdim = _get_dtensor_shard_info(p)
                    global_size = p.shape[sdim]
                    chunk_floor = global_size // ws
                    rem = global_size % ws
                    start = rk * chunk_floor + min(rk, rem)
                    local_size = update_local.shape[sdim]
                    ortho_local = full_ortho.narrow(sdim, start, local_size).contiguous()
                else:
                    ortho_local = batched_newton_schulz(update_local, ns_coefficients, ns_steps, eps)

                lr_shape = p.shape[-2:] if p.ndim >= 2 else p.shape
                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, lr_shape)

                p_local = p.to_local() if isinstance(p, DTensor) else p
                if weight_decay != 0.0:
                    p_local.mul_(1 - lr * weight_decay)
                p_local.add_(ortho_local.to(dtype=p_local.dtype), alpha=-adjusted_lr)

        return loss

    def _build_megabatch_plan(
        self, fsdp_2d_groups: Dict[tuple, List[Tensor]], config: dict
    ) -> List[Dict[str, Any]]:
        """Flatten shape groups into chunks ordered by descending AG bytes.

        Sort keys are global metadata only (shape / dtype / count) => every rank
        builds the identical list and issues the identical collective order.
        Largest-first is load-bearing, not cosmetic: three chunks hold 52.4% of
        the AG bytes and stall a shallow pipeline if left until the end.
        See memory/experiments/plan-muon-p1a-260818.md.
        """
        lr = config["lr"]
        adjust_lr_fn = config["adjust_lr_fn"]
        byte_cap = _ag_byte_cap()
        wire_dtype = _ag_dtype()
        plan: List[Dict[str, Any]] = []

        for key_ord, _key in enumerate(sorted(fsdp_2d_groups.keys())):
            params = fsdp_2d_groups[_key]
            if not params:
                continue
            pg, world_size, rank, shard_dim = _get_dtensor_shard_info(params[0])
            global_shape = tuple(params[0].shape)
            max_local_size = (global_shape[shard_dim] + world_size - 1) // world_size
            # Wire dtype, not param dtype: the cap is a millisecond budget in
            # disguise (bytes / ring goodput), so it must count what is sent.
            itemsize = (wire_dtype or params[0].dtype).itemsize
            per_param = itemsize * max_local_size
            for d, s in enumerate(global_shape):
                if d != shard_dim:
                    per_param *= s
            adjusted_lr = _adjust_lr(lr, adjust_lr_fn, params[0].shape[-2:])

            # P1b: cap the on-wire bytes per chunk so the three ~220 ms giants
            # stop stalling a depth-2 pipeline. Sizes come from global metadata
            # only => identical plan on every rank. 0 keeps the legacy count-only
            # boundaries [32, 32, ..., rem] so cap=0 stays the E1 control exactly.
            # Under a cap the params are spread evenly instead, but never down to
            # a lone param: cuBLAS picks a different baddbmm kernel at batch==1,
            # which breaks bitwise equality for some shapes
            # (scripts/probe_baddbmm_batch1.py). The cap is therefore soft.
            n = len(params)
            if byte_cap:
                gs = max(1, min(_MEGABATCH_MAX_GROUP_SIZE, byte_cap // (world_size * per_param)))
                n_chunk = max(1, min((n + gs - 1) // gs, n // 2))
                base, extra = divmod(n, n_chunk)
                sizes = [base + (i < extra) for i in range(n_chunk)]
            else:
                sizes = [_MEGABATCH_MAX_GROUP_SIZE] * (n // _MEGABATCH_MAX_GROUP_SIZE)
                if n % _MEGABATCH_MAX_GROUP_SIZE:
                    sizes.append(n % _MEGABATCH_MAX_GROUP_SIZE)

            start = 0
            for pos, size in enumerate(sizes):
                chunk = params[start:start + size]
                start += size
                plan.append({
                    "params": chunk,
                    "ag_bytes": world_size * len(chunk) * per_param,
                    "tie": (key_ord, pos),
                    "pg": pg,
                    "world_size": world_size,
                    "rank": rank,
                    "shard_dim": shard_dim,
                    "adjusted_lr": adjusted_lr,
                })

        if _chunk_order() != "legacy":
            plan.sort(key=lambda e: (-e["ag_bytes"], e["tie"]))
        return plan

    def _run_megabatch_pipeline(self, plan: List[Dict[str, Any]], config: dict) -> None:
        """Depth-N pipeline: chunk i+1's all-gather overlaps chunk i's NS.

        ``async_op=True`` moves the AG onto ProcessGroupNCCL's comm stream;
        with the default ``async_op=False`` torch>=2.7 runs it on the *current*
        stream, which is why the baseline's 44 AGs share the compute stream and
        overlap it by exactly 0.00 ms. ``work.wait()`` makes the compute stream
        wait before NS reads the gathered buffer.
        """
        depth = _pipeline_depth()
        inflight: List[Dict[str, Any]] = []

        # NS sharded to chunk owners: one NS per chunk instead of world_size
        # identical ones. Needs a power-of-two world size (the xor schedule is
        # only a perfect matching there); otherwise fall through to the AG path.
        rounds = _ns_shard_rounds()
        if rounds and plan:
            ws = plan[0]["world_size"]
            if ws > 1 and ws & (ws - 1) == 0:
                if not getattr(self, "_ns_shard_logged", False):
                    self._ns_shard_logged = True
                    if plan[0]["rank"] == 0:
                        print(f"[muon] NS_SHARD=1: rounds={rounds} chunks={len(plan)} "
                              f"ws={ws}, staged xor p2p replaces the all-gather", flush=True)
                self._run_ns_shard_pipeline(plan, config, rounds)
                return

        for entry in plan:
            entry["chunk_idx"] = self._muon_chunk_counter
            self._muon_chunk_counter += 1

            if depth == 1:  # legacy fully-serial path, bit-identical schedule
                with _MuonPerfRange(f"muon_group/{entry['chunk_idx']}", self._enable_nvtx):
                    self._megabatch_issue(entry, config, depth)
                    self._megabatch_finish(entry, config, depth)
                continue

            self._megabatch_issue(entry, config, depth)
            inflight.append(entry)
            if len(inflight) >= depth:
                self._megabatch_finish(inflight.pop(0), config, depth)

        while inflight:
            self._megabatch_finish(inflight.pop(0), config, depth)

    @staticmethod
    def _shard_span(global_dim_size: int, world_size: int, rank: int) -> Tuple[int, int]:
        """``(start, size)`` of ``rank``'s slice, matching FSDP2's own split.

        Global metadata only, so a chunk owner can cut every rank's slice out of
        one ortho tensor without asking anybody -- but it must be the SAME split
        FSDP2 used: ``_chunk_with_empty`` = ``torch.chunk`` + empty tail
        (``torch/distributed/fsdp/_fully_shard/_fsdp_common.py:121``), i.e.
        ceil-sized pieces front-loaded and tail ranks possibly empty.  A balanced
        floor+remainder split agrees only while every rank stays non-empty: the
        [55, 768] family matches at ws=8 and splits 4x13,3,0,0 vs 4x7,3x9 at
        ws=16.  See memory/experiments/exp-muon-ns-shard-2n-hang-260828.md.
        """
        chunk = -(-global_dim_size // world_size)
        start = min(rank * chunk, global_dim_size)
        return start, min(chunk, global_dim_size - start)

    def _apply_ortho(
        self, entry: Dict[str, Any], local_ortho_batch: Tensor,
        has_grad: List[bool], config: dict,
    ) -> int:
        """Phase 4: weight decay + ``p -= lr * ortho`` on this rank's slice."""
        lr = config["lr"]
        weight_decay = config["weight_decay"]
        adjusted_lr = entry["adjusted_lr"]
        last_read = -1
        for i, p in enumerate(entry["params"]):
            if not has_grad[i]:
                continue
            p_local = p.to_local() if isinstance(p, DTensor) else p
            if weight_decay != 0.0:
                p_local.mul_(1 - lr * weight_decay)
            p_local.add_(local_ortho_batch[i].to(dtype=p_local.dtype), alpha=-adjusted_lr)
            # Slice ``i`` is last read here; after the highest ``i`` with a grad
            # the buffer is safe to overwrite.
            last_read = i
        return last_read

    def _pack_local(self, entry: Dict[str, Any], config: dict) -> Dict[str, Any]:
        """Phases 1-2 for one chunk: momentum, stack, wire cast, pad to max_local.

        Produces the payload both distribution paths send (all_gather, or the
        owner exchange) plus the metadata needed to undo the padding. Params with
        ``grad=None`` contribute zeros so every rank stays in the collective.
        """
        params = entry["params"]
        world_size = entry["world_size"]
        shard_dim = entry["shard_dim"]
        momentum = config["momentum"]
        nesterov = config["nesterov"]

        # Momentum on LOCAL tensors: DTensor ops here break under torch.compile,
        # which may hand back plain tensors as grads.
        local_updates: List[Tensor] = []
        has_grad: List[bool] = []
        for p in params:
            if p.grad is None:
                p_local = p.to_local() if isinstance(p, DTensor) else p
                local_updates.append(torch.zeros_like(p_local))
                has_grad.append(False)
                continue

            has_grad.append(True)
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            buf = state["momentum_buffer"]

            buf_local = buf.to_local() if isinstance(buf, DTensor) else buf
            grad_local = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad

            buf_local.lerp_(grad_local, 1 - momentum)
            if nesterov:
                update_local = grad_local.lerp(buf_local, momentum)
            else:
                update_local = buf_local.clone()

            local_updates.append(update_local)

        stacked_local = torch.stack(local_updates, dim=0)  # [N, local_M, K]
        del local_updates

        # Downcast the wire payload. Bitwise-safe: NS's first act is the same
        # cast, and cast/cat/pad-with-zeros all commute elementwise.
        wire_dtype = _ag_dtype()
        if wire_dtype is not None and stacked_local.dtype != wire_dtype:
            stacked_local = stacked_local.to(wire_dtype)

        gather_dim = shard_dim + 1  # +1 for the batch dim we prepended
        original_local_size = stacked_local.size(gather_dim)

        # FSDP2 contiguous chunking may give different ranks different local sizes
        # (ceil vs floor when global_dim % world_size != 0), while both dist
        # collectives and the peer exchange want uniform pieces. Pad first.
        global_dim_size = params[0].shape[shard_dim]  # DTensor .shape = global
        max_local_size = (global_dim_size + world_size - 1) // world_size
        if max_local_size != original_local_size:
            pad_amount = max_local_size - original_local_size
            ndim = stacked_local.ndim
            pad_spec = [0] * (2 * ndim)
            pad_spec[2 * (ndim - 1 - gather_dim) + 1] = pad_amount
            stacked_local = torch.nn.functional.pad(stacked_local, pad_spec)

        return {
            "stacked_local": stacked_local,
            "has_grad": has_grad,
            "gather_dim": gather_dim,
            "original_local_size": original_local_size,
            "max_local_size": max_local_size,
            "global_dim_size": global_dim_size,
        }

    @staticmethod
    def _build_ns_rounds(
        plan: List[Dict[str, Any]], world_size: int, rounds: int
    ) -> List[List[List[Dict[str, Any]]]]:
        """LPT the chunks into ``rounds x world_size`` bins; bin (r, o) = owner o.

        Balanced on NS cost, NOT on ``ag_bytes``: after the internal transpose to
        M <= K, NS does 3 matmuls on [B, M, K] per step, so cost/byte scales with
        the short edge and the production shapes spread 5.4x.  Ownership is per
        WHOLE chunk -- splitting one would change the ``baddbmm`` batch dim and
        cuBLAS switches kernels at batch==1, the one thing that breaks bitwise
        equality.  Metadata-only => every rank builds the identical assignment.
        """
        nbin = rounds * world_size
        bins: List[List[Dict[str, Any]]] = [[] for _ in range(nbin)]
        loads = [0.0] * nbin
        for e in sorted(plan, key=lambda e: (-e["ns_cost"], e["tie"])):
            b = min(range(nbin), key=lambda j: (loads[j], j))
            bins[b].append(e)
            loads[b] += e["ns_cost"]
        return [bins[r * world_size:(r + 1) * world_size] for r in range(rounds)]

    @staticmethod
    def _staged_p2p(build, world_size: int, rank: int) -> None:
        """Run ``build(peer)``'s p2p ops one xor matching at a time.

        At step k every rank pairs with ``rank ^ k`` -- a perfect matching for
        k = 1..ws-1 when ws is a power of two, so ws/2 disjoint pairs move at once
        and no two pairs share a link.  Staying serial is load-bearing, not
        conservative.  Measured on this node (zero NVLink, every hop through the
        PCIe root complex), same volume each time:
            1 matching in flight   15.9 GB/s      2 in flight   1.79 GB/s
            all_to_all_single       1.20 GB/s     ring all_gather  12.0 GB/s
        i.e. concurrency collapses it, not topology, and any batched or collective
        form of this traffic is 8-13x slower than the ring it replaces.
        Payloads must stay 16-byte aligned (4-byte alignment costs exactly 2x);
        production trailing dims are all multiples of 8, so this holds by itself.
        See memory/experiments/exp-muon-ns-a2a-*.md, scripts/probe_a2a_align.py.
        """
        for k in range(1, world_size):
            # Empty slices exist once a tail rank gets nothing from torch.chunk;
            # both sides size them from the same global metadata, so dropping the
            # 0-element ops keeps the two op lists matched.
            ops = [o for o in build(rank ^ k) if o.tensor.numel()]
            if ops:
                for work in dist.batch_isend_irecv(ops):
                    work.wait()

    @classmethod
    def _cat_shards(
        cls, shards: List[Tensor], gather_dim: int, global_dim_size: int, world_size: int
    ) -> Tensor:
        """Rank-ordered concat of ws padded shards, per-rank padding stripped."""
        if global_dim_size % world_size == 0:
            return torch.cat(shards, dim=gather_dim)
        real = [
            s.narrow(gather_dim, 0, cls._shard_span(global_dim_size, world_size, r)[1])
            for r, s in enumerate(shards)
        ]
        return torch.cat(real, dim=gather_dim)

    def _run_ns_shard_pipeline(
        self, plan: List[Dict[str, Any]], config: dict, rounds: int
    ) -> None:
        """One Newton-Schulz per chunk instead of ``world_size`` identical copies.

        Today every rank all-gathers every chunk and runs the same NS on it, so
        (ws-1)/ws of that compute is thrown away.  Here each chunk gets one owner:
        the shards travel to the owner, the owner runs NS once, the ws output
        slices travel back.  Both hops use the staged xor exchange (_staged_p2p),
        which is the only p2p pattern that beats the ring on this node.
        Bit-identical to the all-gather path: p2p moves bytes, the ``cat`` is still
        in rank order, NS is untouched, and the slice bounds come from the same
        global metadata as today's ``narrow``.
        Design + measurements: report/_docs/muon_ns_shard_a2a_proposal.md.
        """
        instr = self._enable_nvtx
        pg, world_size, rank = plan[0]["pg"], plan[0]["world_size"], plan[0]["rank"]
        for e in plan:
            if (e["pg"], e["world_size"], e["rank"]) != (pg, world_size, rank):
                raise RuntimeError("NS sharding needs one process group for the step")
            e["ns_cost"] = _ns_cost(tuple(e["params"][0].shape), len(e["params"]))
            e["chunk_idx"] = self._muon_chunk_counter
            self._muon_chunk_counter += 1
        gp = ([dist.get_global_rank(pg, r) for r in range(world_size)]
              if pg is not None else list(range(world_size)))

        for r_idx, owners in enumerate(self._build_ns_rounds(plan, world_size, rounds)):
            mine, flat = owners[rank], []
            for o, row in enumerate(owners):
                for e in row:
                    e["owner"] = o
                    flat.append(e)

            with _MuonPerfRange(f"muon_ns_pack/{r_idx}", instr):
                for e in flat:
                    e["pk"] = pk = self._pack_local(e, config)
                    sl = pk["stacked_local"] = pk["stacked_local"].contiguous()
                    if e["owner"] == rank:
                        e["gl"] = [sl if r == rank else torch.empty_like(sl)
                                   for r in range(world_size)]
                    else:
                        # my slice of the ortho comes back unpadded
                        shape = list(sl.shape)
                        shape[pk["gather_dim"]] = pk["original_local_size"]
                        e["rb"] = torch.empty(shape, dtype=sl.dtype, device=sl.device)

            # -> owner: my shard of every chunk it owns, its shard of every chunk
            #    I own.  Both sides walk the same owner list, so the isend/irecv
            #    order matches without any extra handshake.
            with _MuonPerfRange(f"muon_ns_a2a_fwd/{r_idx}", instr):
                self._staged_p2p(
                    lambda peer: (
                        [dist.P2POp(dist.isend, e["pk"]["stacked_local"], gp[peer], pg)
                         for e in owners[peer]]
                        + [dist.P2POp(dist.irecv, e["gl"][peer], gp[peer], pg) for e in mine]
                    ),
                    world_size, rank,
                )
            for e in flat:
                if e["owner"] != rank:
                    e["pk"]["stacked_local"] = None      # sent, buffer reusable

            with _MuonPerfRange(f"muon_ns_compute/{r_idx}", instr):
                for e in mine:
                    pk = e["pk"]
                    full = self._cat_shards(
                        e["gl"], pk["gather_dim"], pk["global_dim_size"], world_size
                    )
                    e["gl"], pk["stacked_local"] = None, None
                    ortho = batched_newton_schulz(
                        full, config["ns_coefficients"], config["ns_steps"], config["eps"]
                    )
                    del full
                    e["sl"] = [
                        ortho.narrow(pk["gather_dim"],
                                     *self._shard_span(pk["global_dim_size"], world_size, r)
                                     ).contiguous()
                        for r in range(world_size)
                    ]
                    del ortho

            with _MuonPerfRange(f"muon_ns_a2a_rev/{r_idx}", instr):
                self._staged_p2p(
                    lambda peer: (
                        [dist.P2POp(dist.isend, e["sl"][peer], gp[peer], pg) for e in mine]
                        + [dist.P2POp(dist.irecv, e["rb"], gp[peer], pg) for e in owners[peer]]
                    ),
                    world_size, rank,
                )

            with _MuonPerfRange(f"muon_ns_apply/{r_idx}", instr):
                for e in flat:
                    lob = e["sl"][rank] if e["owner"] == rank else e["rb"]
                    self._apply_ortho(e, lob, e["pk"]["has_grad"], config)
                    e["params"] = e["pk"] = e["sl"] = e["rb"] = None

    def _megabatch_issue(self, entry: Dict[str, Any], config: dict, depth: int) -> None:
        """Phases 1-2 for one chunk: momentum + stack/pad, then launch the AG.

        ``async_op=depth > 1`` is the whole point of P1a-ii: it puts the
        all-gather on ProcessGroupNCCL's comm stream instead of the current
        (compute) stream, which is what lets it overlap the previous chunk's
        Newton-Schulz. Emits ``muon_pack/{i}`` + ``muon_ag/{i}``.
        """
        params = entry["params"]
        world_size = entry["world_size"]
        shard_dim = entry["shard_dim"]
        chunk_idx = entry["chunk_idx"]
        instr = self._enable_nvtx

        # Per-chunk stats scaffold — filled progressively; timings resolved
        # lazily in ``dump_stats`` after a global CUDA sync.
        stat: Dict[str, Any] = {
            "chunk_idx": chunk_idx,
            "shape": tuple(params[0].shape) if params else (),
            "dtype": str(params[0].dtype) if params else "",
            "param_count": len(params),
            "with_grad_count": 0,
            "world_size": int(world_size),
            "shard_dim": int(shard_dim),
            "local_size": 0,
            "max_local_size": 0,
            "ag_in_bytes": 0,
            "ag_out_bytes": 0,
            "pipeline_depth": int(depth),
            "pack_ms": None,
            "ag_ms": None,
            "wait_ms": None,
            "ns_ms": None,
            "apply_ms": None,
            "last_read_iter": -1,
        }
        entry["stat"] = stat
        ev: Dict[str, Any] = {}
        entry["ev"] = ev
        if instr:
            for _ph in ("pack", "ag", "ns", "apply"):
                ev[f"{_ph}_s"], ev[f"{_ph}_e"] = self._make_event_pair()
            # ``wait_ms`` = the EXPOSED part of the AG (wait_s -> AG complete).
            # It is what must collapse when the pipeline works; ``ag_ms``
            # (issue -> complete) also spans other chunks' compute by design.
            ev["wait_s"] = torch.cuda.Event(enable_timing=True)
            ev["wait_e"] = ev["ag_e"]

        with _MuonPerfRange(f"muon_pack/{chunk_idx}", instr):
            if instr:
                ev["pack_s"].record()
            packed = self._pack_local(entry, config)
            stacked_local = packed["stacked_local"]
            has_grad = packed["has_grad"]
            gather_dim = packed["gather_dim"]
            original_local_size = packed["original_local_size"]
            max_local_size = packed["max_local_size"]
            global_dim_size = packed["global_dim_size"]
            if instr:
                ev["pack_e"].record()
                stat["with_grad_count"] = int(sum(has_grad))
                stat["local_size"] = int(original_local_size)
                stat["max_local_size"] = int(max_local_size)

        with _MuonPerfRange(f"muon_ag/{chunk_idx}", instr):
            if instr:
                ev["ag_s"].record()
            gather_list = [torch.empty_like(stacked_local) for _ in range(world_size)]
            if instr:
                stat["ag_in_bytes"] = int(stacked_local.numel() * stacked_local.element_size())
                # gather_list buffers include this rank's own slot (== ag_in),
                # matching the on-wire AG payload accounting.
                stat["ag_out_bytes"] = int(
                    world_size * stacked_local.numel() * stacked_local.element_size()
                )
            stacked_local = stacked_local.contiguous()
            work = dist.all_gather(
                gather_list, stacked_local, group=entry["pg"], async_op=depth > 1
            )

        # Handed to _megabatch_finish; ``stacked_local`` must stay alive until
        # the AG completes, so it is kept referenced here rather than deleted.
        entry.update({
            "work": work,
            "gather_list": gather_list,
            "stacked_local": stacked_local,
            "has_grad": has_grad,
            "gather_dim": gather_dim,
            "original_local_size": original_local_size,
            "max_local_size": max_local_size,
            "global_dim_size": global_dim_size,
        })

    def _megabatch_finish(self, entry: Dict[str, Any], config: dict, depth: int) -> None:
        """Wait for the chunk's AG, then Phases 3-4 (NS + scatter/apply).

        ``work.wait()`` only makes the current stream wait on the comm stream;
        it does not block the CPU, so the next chunk's AG is already in flight.
        Emits ``muon_ns/{i}`` + ``muon_apply/{i}``.
        """
        params = entry["params"]
        world_size = entry["world_size"]
        chunk_idx = entry["chunk_idx"]
        stat, ev = entry["stat"], entry["ev"]
        instr = self._enable_nvtx
        gather_dim = entry["gather_dim"]
        global_dim_size = entry["global_dim_size"]
        max_local_size = entry["max_local_size"]
        original_local_size = entry["original_local_size"]
        has_grad = entry["has_grad"]

        with _MuonPerfRange(f"muon_wait/{chunk_idx}", instr):
            if instr:
                ev["wait_s"].record()
            if entry["work"] is not None:
                entry["work"].wait()
            entry["work"] = None
            entry["stacked_local"] = None  # AG done => input buffer reusable

            gather_list = entry["gather_list"]
            entry["gather_list"] = None
            stacked_full = self._cat_shards(
                gather_list, gather_dim, global_dim_size, world_size
            )
            del gather_list
            if instr:
                ev["ag_e"].record()

        with _MuonPerfRange(f"muon_ns/{chunk_idx}", instr):
            if instr:
                ev["ns_s"].record()
            # Phase 3: Batched Newton-Schulz
            stacked_ortho = batched_newton_schulz(
                stacked_full, config["ns_coefficients"], config["ns_steps"], config["eps"]
            )
            del stacked_full
            if instr:
                ev["ns_e"].record()

        with _MuonPerfRange(f"muon_apply/{chunk_idx}", instr):
            if instr:
                ev["apply_s"].record()
            # Phase 4: Local scatter + apply update
            shard_start, _ = self._shard_span(global_dim_size, world_size, entry["rank"])
            local_ortho_batch = stacked_ortho.narrow(
                gather_dim, shard_start, original_local_size
            ).contiguous()
            del stacked_ortho
            last_read = self._apply_ortho(entry, local_ortho_batch, has_grad, config)

            if instr:
                ev["apply_e"].record()
                stat["last_read_iter"] = int(last_read)

        entry["params"] = None
        if instr:
            self._muon_stats.append(stat)
            self._muon_events.append(ev)

    @staticmethod
    def _compute_ortho(
        update: Tensor,
        kind: str,
        ns_coefficients: Tuple[float, float, float],
        ns_steps: int,
        eps: float,
    ) -> Tensor:
        """Run Newton-Schulz on ``update`` according to its layout kind."""
        if kind == _KIND_LOCAL:
            return batched_newton_schulz(update, ns_coefficients, ns_steps, eps)

        if kind == _KIND_FSDP_GATHER_2D:
            full = _full_grad(update)
            return batched_newton_schulz(full, ns_coefficients, ns_steps, eps)

        if kind == _KIND_MOE_LOCAL_3D:
            assert isinstance(update, DTensor)
            local = update._local_tensor
            local_ortho = batched_newton_schulz(local, ns_coefficients, ns_steps, eps)
            return DTensor.from_local(
                local_ortho,
                device_mesh=update.device_mesh,
                placements=update.placements,
                run_check=False,
            )

        if kind == _KIND_MOE_GATHER_3D:
            full = _full_grad(update)
            return batched_newton_schulz(full, ns_coefficients, ns_steps, eps)

        raise ValueError(f"Unknown DistributedMuon kind: {kind!r}")