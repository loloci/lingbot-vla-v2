"""Verify the column-first gather is bit-identical to row-first, on ALL sub-datasets.

Read-only: no model, no CUDA, no writes. Correctness only -- timing lives in
tools/probe_dataloader_cost.py.

Why this exists
---------------
The probe proved the row-first gather at base_dataset.py:99 costs 1487 ms/sample
and that a one-column view returns the same values ~1500x faster -- but it only
checked sub-dataset 0 (adjust_bottle) on random interior indices. Before changing
production code the equality has to hold on all 5 sub-datasets, and at episode
boundaries where _get_query_indices clamps and emits REPEATED indices.

What could make row-first right and the view wrong -- i.e. why this script exists
rather than a shrug:

  1. select_columns is @transmit_format-decorated, and that wrapper RECOMPUTES
     the format column list as
         sorted(set(view.column_names) - (all_columns - self._format_columns))
     If _format_columns were a strict subset NOT containing the gathered key, the
     view's format columns come out EMPTY and the value is no longer a tensor.
     Sub-dataset 0 has all 10 columns formatted; the other 4 are asserted here.

  2. hf_transform_to_torch picks its branch from items_dict[key][0], including an
     `elif first_item is None: pass` path that leaves a column as raw pylist. The
     decision is per-key so it cannot differ between the two forms for the same
     row set -- but that is an argument, and this script turns it into a check.

  3. An _indices mapping (left by .select()/.shuffle()) must survive
     copy.deepcopy inside select_columns, or view row i is not parent row i.

  4. Clamped/repeated indices at episode start and end.

  5. Feature dtype: bf16/float32/uint8 differences would show up as dtype
     mismatch rather than value mismatch, so shape and dtype are compared too.

The strongest check here is not the gather comparison but check_items(): it runs
the REAL VLADataset.__getitem__ twice -- once with production's own
_query_hf_dataset, once with the other form monkey-patched in -- and compares
every tensor in the returned dict, after feature_transform. If those are
bit-identical the change is invisible to training by construction.

Both gathers are re-implemented standalone here (gather_row_first /
gather_col_first, with the view cached on `lr._verify_views`), so this script
tests the two forms against each other regardless of which one production
currently uses. It was written before the change landed at base_dataset.py:99 and
stays valid after: rerun it if that line, `select_columns`, or the `datasets`
version ever moves.

Usage (inside the training container, single process):
    python tools/verify_column_gather.py <config.yaml> [--data.num_workers 0]

Env knobs:
    VERIFY_N          random interior indices per sub-dataset   (default 8)
    VERIFY_EPISODES   episodes whose boundaries are probed      (default 3)
    VERIFY_ITEMS      full-item comparisons per sub-dataset     (default 4)
"""
from __future__ import annotations

import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE))

from probe_dataloader_cost import build_dataset  # noqa: E402  (shares the arg plumbing)

FAILURES: list[str] = []


def _p(*a):
    print(*a, flush=True)


def _fail(msg):
    FAILURES.append(msg)
    _p(f"  [FAIL] {msg}")


def col_view(lr, key):
    """The proposed implementation, standalone so nothing in the repo is patched."""
    views = getattr(lr, "_verify_views", None)
    if views is None:
        views = lr._verify_views = {}
    if key not in views:
        views[key] = lr.hf_dataset.select_columns([key])
    return views[key]


def gather_row_first(lr, key, rel):
    return torch.stack(lr.hf_dataset[rel][key])


def gather_col_first(lr, key, rel):
    return torch.stack(col_view(lr, key)[rel][key])


def check_format_inheritance(lr, tag):
    """Guard #1: the view must keep the custom transform, for EVERY gathered key.

    transmit_format recomputes the format column list from a set difference, so a
    parent that formats only a subset of columns can hand the view an empty list.
    That would silently return raw pylists instead of tensors.
    """
    hf = lr.hf_dataset
    pf = hf.format
    all_cols = set(hf.column_names)
    fmt_cols = set(pf["columns"] or [])
    _p(f"  parent format: type={pf['type']!r} columns={len(fmt_cols)}/{len(all_cols)} "
       f"output_all_columns={pf['output_all_columns']}")
    if fmt_cols != all_cols:
        _p(f"  [WARN] parent formats a SUBSET; unformatted = {sorted(all_cols - fmt_cols)}")

    keys = [k for k in (lr.delta_indices or {}) if k not in lr.meta.video_keys]
    for key in keys:
        v = col_view(lr, key)
        vf = v.format
        if vf["type"] != pf["type"]:
            _fail(f"{tag} {key}: view format type {vf['type']!r} != parent {pf['type']!r}")
        if vf["format_kwargs"].get("transform") is not pf["format_kwargs"].get("transform"):
            _fail(f"{tag} {key}: view lost the hf_transform_to_torch transform")
        if key not in (vf["columns"] or []):
            _fail(f"{tag} {key}: key not in view format columns {vf['columns']} -> would return raw pylist")
        if v.num_rows != lr.hf_dataset.num_rows:
            _fail(f"{tag} {key}: view num_rows {v.num_rows} != parent {lr.hf_dataset.num_rows}")
        if v._indices is not lr.hf_dataset._indices:
            # deepcopy of an _indices table is fine as long as it still maps the
            # same rows; equality of the row it resolves to is checked below.
            same = (v._indices is None) == (lr.hf_dataset._indices is None)
            _p(f"  note {key}: view._indices is a copy (both-None={same})")


def _episode_bounds(lr, ep):
    """[from, to) row range of episode `ep`, across both LeRobot dataset APIs.

    v2 exposes `episode_data_index['from'/'to']`; v3 dropped that attribute and
    keeps the same numbers under `meta.episodes[ep]['dataset_from_index'/
    '_to_index']` (the split this repo already handles at
    scripts/open_loop_eval.py:207-210). Returning None means "cannot locate", and
    the caller degrades to interior-only sampling -- but says so loudly, because
    silently skipping the boundary cases is exactly how the clamp/repeat hazard
    would go unchecked.
    """
    edi = getattr(lr, "episode_data_index", None)
    if edi is not None:
        return int(edi["from"][ep].item()), int(edi["to"][ep].item())
    ep_row = lr.meta.episodes[ep]
    return int(ep_row["dataset_from_index"]), int(ep_row["dataset_to_index"])


def check_gathers(lr, tag, n_random, n_episodes, seed):
    """Compare the two gathers on random interior indices AND episode boundaries."""
    from lingbotvla.data.vla_data.base_dataset import _to_relative_indices

    rng = random.Random(seed)
    di = lr.delta_indices or {}
    keys = [k for k in di if k not in lr.meta.video_keys]
    n = lr.hf_dataset.num_rows

    cases: list[tuple[str, int]] = []
    for _ in range(n_random):
        cases.append(("interior", rng.randrange(n)))

    # Episode boundaries: _get_query_indices clamps to the episode, producing
    # repeated indices. Those repeats are exactly where a subtle indexing bug
    # would show up, and random interior sampling only hits them by luck.
    n_ep = min(n_episodes, int(lr.meta.total_episodes))
    n_bounds = 0
    for ep in range(n_ep):
        try:
            frm, to = _episode_bounds(lr, ep)
        except Exception as exc:  # noqa: BLE001
            _fail(f"{tag}: episode bounds unavailable ({type(exc).__name__}: {exc}) "
                  f"-- boundary cases NOT covered")
            break
        cases += [("ep_start", frm), ("ep_start+1", frm + 1), ("ep_end-1", to - 2), ("ep_end", to - 1)]
        n_bounds += 4
    _p(f"  boundary indices probed: {n_bounds} (from {n_ep} episodes)")

    n_cmp = 0
    n_repeat_cases = 0
    for kind, idx in cases:
        if not (0 <= idx < n):
            continue
        ep_idx = int(lr.hf_dataset[idx]["episode_index"].item())
        q, _pad = lr._get_query_indices(idx, ep_idx)
        for key in keys:
            if key not in q:
                continue
            rel = _to_relative_indices(lr, q[key])
            if len(set(rel)) != len(rel):
                n_repeat_cases += 1
            a = gather_row_first(lr, key, rel)
            b = gather_col_first(lr, key, rel)
            n_cmp += 1
            if a.shape != b.shape:
                _fail(f"{tag} {kind} idx={idx} {key}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
            elif a.dtype != b.dtype:
                _fail(f"{tag} {kind} idx={idx} {key}: dtype {a.dtype} vs {b.dtype}")
            elif not torch.equal(a, b):
                d = (a.float() - b.float()).abs().max().item()
                _fail(f"{tag} {kind} idx={idx} {key}: VALUES differ, max|diff|={d:g}")
    _p(f"  gathers compared: {n_cmp}  (of which {n_repeat_cases} had clamped/repeated indices)")


def _compare_items(a, b, tag):
    """Compare two full __getitem__ outputs key by key."""
    if set(a) != set(b):
        _fail(f"{tag}: key sets differ, only-A={sorted(set(a)-set(b))} only-B={sorted(set(b)-set(a))}")
        return
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, torch.Tensor) != isinstance(vb, torch.Tensor):
            _fail(f"{tag} {k}: type {type(va).__name__} vs {type(vb).__name__}")
        elif isinstance(va, torch.Tensor):
            if va.shape != vb.shape:
                _fail(f"{tag} {k}: shape {tuple(va.shape)} vs {tuple(vb.shape)}")
            elif va.dtype != vb.dtype:
                _fail(f"{tag} {k}: dtype {va.dtype} vs {vb.dtype}")
            elif not torch.equal(va, vb):
                d = (va.float() - vb.float()).abs().max().item()
                _fail(f"{tag} {k}: VALUES differ, max|diff|={d:g}")
        elif va != vb:
            _fail(f"{tag} {k}: {va!r} vs {vb!r}")


def check_items(ds, sub, lr, tag, n_items, seed):
    """The decisive check: run the REAL __getitem__ both ways and diff everything.

    Monkey-patches _query_hf_dataset on the instance with the ROW-FIRST form for
    the second pass, then restores it. Patching in row-first rather than
    column-first keeps this check meaningful in BOTH directions: before the change
    it compares production(row) vs column, after the change production(column) vs
    row. Patching in column-first would go degenerate once the change lands.

    This covers feature_transform.apply, normalization, tokenization and
    prepare_images -- i.e. it answers "is the change invisible to training?"
    rather than "is one gather equal to another?".
    """
    from lingbotvla.data.vla_data.base_dataset import _to_relative_indices

    rng = random.Random(seed + 7)
    total = len(sub)
    start = getattr(sub, "dataset_start_index", 0)

    def patched(query_indices):
        out = {}
        for key, q_idx in query_indices.items():
            if key in lr.meta.video_keys:
                continue
            out[key] = gather_row_first(lr, key, _to_relative_indices(lr, q_idx))
        return out

    orig = lr._query_hf_dataset
    n_done = 0
    for _ in range(n_items):
        local = rng.randrange(total)
        try:
            a = sub[local]
            lr._query_hf_dataset = patched
            try:
                b = sub[local]
            finally:
                lr._query_hf_dataset = orig
        except Exception as exc:  # noqa: BLE001
            _fail(f"{tag} item local={local}: raised {type(exc).__name__}: {exc}")
            lr._query_hf_dataset = orig
            continue
        _compare_items(a, b, f"{tag} item local={local}")
        n_done += 1
    _p(f"  full items compared: {n_done}  (start_index={start}, sub len={total})")


def main():
    if len(sys.argv) < 2:
        _p("usage: python tools/verify_column_gather.py <config.yaml>")
        sys.exit(1)

    torch.set_num_threads(int(os.environ.get("VERIFY_THREADS", 4)))
    n_random = int(os.environ.get("VERIFY_N", 8))
    n_episodes = int(os.environ.get("VERIFY_EPISODES", 3))
    n_items = int(os.environ.get("VERIFY_ITEMS", 4))

    t0 = time.time()
    args, ds = build_dataset()
    subs = ds._datasets if hasattr(ds, "_datasets") else [ds]
    _p(f"[verify] dataset built in {time.time()-t0:.1f} s: {len(subs)} sub-datasets, {len(ds)} samples")
    _p(f"[verify] per sub-dataset: {n_random} interior idx, {n_episodes} episodes x 4 boundary idx, "
       f"{n_items} full items")

    for i, sub in enumerate(subs):
        lr = sub.dataset
        _p("")
        _p(f"[sub {i}] {getattr(lr, 'repo_id', '?')}  rows={lr.hf_dataset.num_rows} "
           f"episodes={lr.meta.total_episodes}")
        _p(f"  video_keys={list(lr.meta.video_keys)} camera_keys={list(lr.meta.camera_keys)}")
        check_format_inheritance(lr, f"sub{i}")
        check_gathers(lr, f"sub{i}", n_random, n_episodes, seed=1234 + i)
        check_items(ds, sub, lr, f"sub{i}", n_items, seed=1234 + i)

    _p("")
    _p("=" * 68)
    if FAILURES:
        _p(f"[verify] RESULT: {len(FAILURES)} FAILURE(S) -- do NOT change base_dataset.py")
        for f in FAILURES[:40]:
            _p(f"  - {f}")
    else:
        _p("[verify] RESULT: PASS -- column-first is bit-identical on all sub-datasets,")
        _p("         including episode boundaries with clamped/repeated indices,")
        _p("         and full __getitem__ outputs match after feature_transform.")
    _p("=" * 68)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
