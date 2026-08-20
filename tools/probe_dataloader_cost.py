"""Measure the per-sample cost of the training dataloader, and A/B the row-first
vs column-first index gather in LeRobotDataset._query_hf_dataset.

Read-only: builds no model, allocates no GPU memory, touches no checkpoint, and
never mutates the dataset on disk. Safe to run alongside nothing / anything.

Why this exists
---------------
The periodic step-time anomaly is a dataloader *supply-rate* ceiling: end-to-end
step time is pinned at ~9.5-10.4 s across every configuration while pure
compute+comm ranges 5.90-8.94 s. From the lag-8 closure (7C + J = 8 x mean, 99.8%
on base 1n) one worker needs T_prod = 77.9-80.5 s to build one 48-sample batch,
i.e. 1.62-1.68 s per sample. This probe measures that number directly and splits
it, so the fix is chosen from data instead of from a guess.

The specific suspicion under test, base_dataset.py:81-100 --

    \"\"\"Tries column-first [key][indices] for speed, falls back to row-first.\"\"\"
    ...
    result[key] = torch.stack(self.hf_dataset[relative_indices][key])

The docstring says column-first; the code is row-first. HF datasets materializes
*whole rows*, so every gather also decodes the 3 PNG image columns of every row
it touches -- then throws them away for non-image keys. For this dataset:

    delta_timestamps = {'action': 50 timestamps}                  (chunk_size=50)
                     u {3 camera keys: [0, (chunk_size-1)/fps]}   (use_future_image)

and meta.video_keys is EMPTY (info.json: video features 0, image features 3), so
the `key in self.meta.video_keys` skip at :95 never fires and the camera keys go
down the same row-first path. Predicted decode count per sample:

    hf_dataset[idx]              1 row  x 3 images =   3   (needed: current frame)
    'action'      50 rows x 3 images = 150             (needed: 0)
    3 cam keys     2 rows x 3 images =  18  ( x3 keys)  (needed: 6, one per cam)
    -------------------------------------------------------------------
    total ~171 decodes per sample, of which ~9 are needed.

If that holds, the fix is to gather one column instead of whole rows. NOTE the
literal form in the :85 docstring (`hf_dataset[key][indices]`) does NOT work in
datasets 3.6.0 -- `hf[key]` materializes the whole formatted column and a list
cannot index a list. The two forms that do work, both measured here and both
required to be bit-identical to the current output before either is proposed:
    select_columns   hf.select_columns([key])[indices][key]   (view built once)
    arrow_take       hf.data.column(key).slice(...) / [i].as_py()  (non-image only;
                     ChunkedArray.take() overflows int32 offsets on the image
                     columns because it concatenates all 494 chunks first)

Usage (inside the training container, single process, no torchrun):
    python tools/probe_dataloader_cost.py configs/vla/robotwin/robotwin.yaml

Env knobs:
    PROBE_N        samples to time            (default 24)
    PROBE_WARMUP   untimed samples first      (default 3)
    PROBE_SEED     index RNG seed             (default 1234)
    PROBE_THREADS  torch.set_num_threads(...) (default: leave alone)
"""
from __future__ import annotations

import json
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

def _p(*a):
    print(*a, flush=True)


def _stats(name, xs):
    if not xs:
        return
    xs_sorted = sorted(xs)
    _p(
        f"  {name:<26} n={len(xs):<4} mean={statistics.mean(xs)*1e3:9.2f} ms  "
        f"median={statistics.median(xs)*1e3:9.2f} ms  "
        f"min={xs_sorted[0]*1e3:9.2f}  max={xs_sorted[-1]*1e3:9.2f}"
    )


def build_dataset():
    """Build the real training dataset with no foundation model.

    The FeatureTransform needs a model config for max_state_dim / max_action_dim,
    so the config class is built from the registry rather than by loading 6B of
    weights.
    """
    _tasks_dir = _REPO / "tasks" / "vla"
    sys.path.insert(0, str(_tasks_dir))
    from train_lingbotvla import Arguments  # type: ignore

    from lingbotvla.utils.arguments import parse_args

    args = parse_args(Arguments)
    args.train.local_rank = 0
    args.train.world_size = 1
    args.train.global_rank = 0

    from lingbotvla.data import build_vla_dataset
    from lingbotvla.models import build_processor
    from lingbotvla.models.config_registry import get_config_registry

    processor = build_processor(args.model.tokenizer_path)
    registry = get_config_registry()
    config_kwargs = {**vars(args.model), **vars(args.train)}
    ConfigCls = registry.get_config_cls_from_config_key(args.model.config_key)
    fake_model_config = ConfigCls(**config_kwargs)

    args.data.chunk_size = args.train.chunk_size
    use_depth_align = args.train.align_params != {}
    ds = build_vla_dataset(
        dataset_config=args.data,
        model_config=args.model,
        config=fake_model_config,
        processor=processor,
        use_depth_align=use_depth_align,
    )
    return args, ds

def describe_layout(ds):
    """Print what the delta-index plan actually is, so the decode count is fact."""
    sub = ds._datasets[0] if hasattr(ds, "_datasets") else ds
    lr = sub.dataset  # LeRobotDataset
    _p("")
    _p("[layout] ---------------------------------------------------------------")
    _p(f"  sub-datasets           : {len(ds._datasets) if hasattr(ds, '_datasets') else 1}")
    _p(f"  total samples          : {len(ds)}")
    _p(f"  meta.video_keys        : {list(lr.meta.video_keys)}")
    _p(f"  meta.camera_keys       : {list(lr.meta.camera_keys)}")
    _p(f"  chunk_size             : {sub.chunk_size}")
    _p(f"  use_future_image       : {sub.use_future_image}")
    di = getattr(lr, "delta_indices", None)
    if di is None:
        _p("  delta_indices          : None")
    else:
        total_rows = 0
        for k, v in di.items():
            total_rows += len(v)
            _p(f"    delta_indices[{k!r}] : {len(v)} offsets")
        _p(f"  total gathered rows/sample (excl. the base row) : {total_rows}")
        n_img = len(lr.meta.camera_keys)
        _p(
            f"  => row-first materializes ~{(total_rows + 1) * n_img} image decodes/sample, "
            f"of which ~{1 * n_img + (n_img if sub.use_future_image else 0)} are used"
        )
    hf = lr.hf_dataset
    _p(f"  hf_dataset             : {type(hf).__name__}  num_rows={hf.num_rows}")
    _p(f"  hf_dataset.format      : {hf.format}")
    _p(f"  column_names           : {hf.column_names}")
    _p("[layout] ---------------------------------------------------------------")
    return sub, lr


def probe_gather_variants(lr, n_trials, seed):
    """A/B the ways to fetch delta-indexed columns, on the real dataset.

    Correctness first: variants must return values equal to the current code's,
    otherwise the timing is meaningless. Only then report timings.

    On what "column-first" must mean here. The docstring at base_dataset.py:85
    says `[key][indices]`, but in datasets 3.6.0 `hf[key]` materializes the WHOLE
    formatted column, so that literal form is worse, not better. The two forms
    that actually work:
      select_columns  a one-column *view*, built once, then indexed by row. The
                      view has no image columns, so row formatting decodes nothing.
      arrow_take      go under `datasets` to the Arrow table and read the column
                      directly. Fastest, but bypasses hf_transform_to_torch, so
                      dtype/shape must be asserted equal, not assumed -- and it is
                      only applicable to non-image columns.
    """
    from lingbotvla.data.vla_data.base_dataset import _to_relative_indices

    di = getattr(lr, "delta_indices", None)
    if not di:
        _p("[gather] delta_indices is empty — nothing to A/B")
        return

    hf = lr.hf_dataset
    rng = random.Random(seed)
    n = hf.num_rows
    keys = [k for k in di.keys() if k not in lr.meta.video_keys]
    span = max(max(v) for v in di.values()) - min(min(v) for v in di.values())
    lo = abs(min(min(v) for v in di.values()))

    _p("")
    _p(f"[gather] A/B on keys={keys}  span={span}  trials={n_trials}  rows={n}")

    # Build the one-column views once, and time that build: if it were expensive
    # it would have to be amortized at dataset __init__, which changes the fix.
    views = {}
    t0 = time.perf_counter()
    for key in keys:
        views[key] = hf.select_columns([key])
    _p(
        f"[gather] select_columns view build, {len(keys)} keys: "
        f"{(time.perf_counter()-t0)*1e3:.2f} ms (one-time, not per sample)"
    )

    timings: dict[str, list[float]] = {"row_first (current)": [], "select_columns": [], "arrow_take": []}
    # Per-key timings. The aggregate medians mix a 50-row gather with three 2-row
    # gathers and understate the effect, so keep them split: the projected saving
    # per sample is sum-over-keys, not n_keys x median.
    per_key: dict[str, dict[str, list[float]]] = {k: {} for k in keys}
    mismatch_sc = 0
    mismatch_at = 0
    # arrow_take is only meaningful for non-image columns (see comment below).
    arrow_keys = {k for k in keys if k not in lr.meta.camera_keys}
    _p(f"[gather] arrow_take applies to {sorted(arrow_keys)} (image columns excluded: raw PNG bytes)")

    def _rec(key, variant, dt):
        timings[variant].append(dt)
        per_key[key].setdefault(variant, []).append(dt)

    for _ in range(n_trials):
        base = rng.randrange(lo, max(lo + 1, n - span - 1))
        for key in keys:
            q_idx = [base + off for off in di[key]]
            rel = _to_relative_indices(lr, q_idx)
            if not rel or max(rel) >= n or min(rel) < 0:
                continue

            t0 = time.perf_counter()
            ref = torch.stack(hf[rel][key])
            _rec(key, "row_first (current)", time.perf_counter() - t0)

            t0 = time.perf_counter()
            got = torch.stack(views[key][rel][key])
            _rec(key, "select_columns", time.perf_counter() - t0)
            if not (got.shape == ref.shape and got.dtype == ref.dtype and torch.equal(ref, got)):
                mismatch_sc += 1

            # arrow_take: only for non-image keys.
            #   - Image columns hold raw PNG bytes, so Arrow returns
            #     {'bytes','path'} dicts, not the decoded tensors the current code
            #     returns -- not comparable, and not what we'd want anyway.
            #   - ChunkedArray.take() concatenates all 494 chunks first, which
            #     overflows int32 binary offsets on the multi-GB image columns
            #     (ArrowInvalid: offset overflow). Slice/scalar access does not
            #     concatenate, so use those.
            if key in arrow_keys:
                t0 = time.perf_counter()
                col = hf.data.column(key)
                if rel == list(range(rel[0], rel[0] + len(rel))):
                    vals = col.slice(rel[0], len(rel)).to_pylist()
                else:
                    vals = [col[i].as_py() for i in rel]
                got2 = torch.stack([torch.as_tensor(v) for v in vals])
                _rec(key, "arrow_take", time.perf_counter() - t0)
                if not (got2.shape == ref.shape and got2.dtype == ref.dtype and torch.equal(ref, got2)):
                    mismatch_at += 1

    _p(f"[gather] mismatches vs current: select_columns={mismatch_sc}  arrow_take={mismatch_at}")
    _p("[gather] (each MUST be 0 for that variant's timing to mean anything)")
    _p("[gather] aggregate over all keys (mixes 50-row and 2-row gathers):")
    for k, v in timings.items():
        _stats(k, v)

    _p("")
    _p("[gather] per key --------------------------------------------------")
    saved_sc = 0.0
    saved_best = 0.0
    cur_total = 0.0
    for key in keys:
        if "row_first (current)" not in per_key[key]:
            continue
        _p(f"  {key}  ({len(di[key])} rows/gather)")
        for variant in ("row_first (current)", "select_columns", "arrow_take"):
            if variant in per_key[key]:
                _stats("    " + variant, per_key[key][variant])
        rf = statistics.median(per_key[key]["row_first (current)"])
        cur_total += rf
        sc = statistics.median(per_key[key]["select_columns"])
        saved_sc += rf - sc
        best = min(sc, statistics.median(per_key[key]["arrow_take"])) if "arrow_take" in per_key[key] else sc
        saved_best += rf - best
        line = f"    => select_columns {rf/sc:7.1f}x"
        if "arrow_take" in per_key[key]:
            at = statistics.median(per_key[key]["arrow_take"])
            line += f"   arrow_take {rf/at:7.1f}x"
        _p(line)

    _p("")
    _p(f"  current delta-gather total / sample   {cur_total*1e3:9.2f} ms  (sum of per-key medians)")
    _p(f"  saving with select_columns everywhere {saved_sc*1e3:9.2f} ms")
    _p(f"  saving with best variant per key      {saved_best*1e3:9.2f} ms")
    _p("[gather] ---------------------------------------------------------")

def probe_stages(ds, sub, lr, n, warmup, seed):
    """Time the real __getitem__ and its three internal stages.

    Stage split follows LeRobotDataset.__getitem__ (base_dataset.py:123-150):
      base row       hf_dataset[idx]
      delta gather   _query_hf_dataset(query_indices)
      Resize         image_transforms per camera key
    Then VLADataset.getitem adds feature_transform.apply (normalize + tokenize +
    prepare_images), which is timed as the remainder.
    """
    from lingbotvla.data.vla_data.base_dataset import _to_relative_indices

    rng = random.Random(seed)
    total = len(ds)
    t_full, t_base, t_delta, t_resize = [], [], [], []

    for i in range(warmup + n):
        idx = rng.randrange(total)
        timed = i >= warmup

        t0 = time.perf_counter()
        item = ds[idx]
        dt_full = time.perf_counter() - t0
        del item

        # Re-run the internals on the same index. Page cache is warm from the
        # full call above, so these are lower bounds on the cold cost — the point
        # is the *ratio* between stages, not their absolute floor.
        local = rng.randrange(len(lr))
        t0 = time.perf_counter()
        row = lr.hf_dataset[local]
        dt_base = time.perf_counter() - t0

        ep_idx = row["episode_index"].item()
        t0 = time.perf_counter()
        qi, _pad = lr._get_query_indices(local, ep_idx)
        _ = lr._query_hf_dataset(qi)
        dt_delta = time.perf_counter() - t0

        t0 = time.perf_counter()
        if lr.image_transforms is not None and lr.load_image:
            for cam in lr.meta.camera_keys:
                _ = lr.image_transforms(row[cam])
        dt_resize = time.perf_counter() - t0

        if timed:
            t_full.append(dt_full)
            t_base.append(dt_base)
            t_delta.append(dt_delta)
            t_resize.append(dt_resize)

    _p("")
    _p("[stages] per-sample cost --------------------------------------------")
    _stats("full ds[idx] (end to end)", t_full)
    _stats("  hf_dataset[idx] base row", t_base)
    _stats("  _query_hf_dataset delta", t_delta)
    _stats("  image_transforms Resize", t_resize)
    acc = statistics.mean(t_base) + statistics.mean(t_delta) + statistics.mean(t_resize)
    full = statistics.mean(t_full)
    _p(f"  accounted            {acc*1e3:9.2f} ms of {full*1e3:9.2f} ms  ({100*acc/full:5.1f}%)")
    _p(f"  remainder (feature_transform.apply + collate-side) {(full-acc)*1e3:9.2f} ms")
    return full


def report_supply_math(per_sample_s, micro_batch_size, num_workers):
    """Turn a measured per-sample cost into the supply-ceiling numbers.

    num_workers comes from the config the probe was launched with, which the
    driver overrides to 0 (single process, no worker pool). So report the
    production value too -- the ceiling is a property of the training config,
    not of this probe.
    """
    t_prod = per_sample_s * micro_batch_size
    _p("")
    _p("[supply] -----------------------------------------------------------")
    _p(f"  measured per-sample            {per_sample_s:8.3f} s")
    _p(f"  micro_batch_size               {micro_batch_size}")
    _p(f"  => T_prod (1 batch, 1 worker)  {t_prod:8.1f} s")
    _p(f"  num_workers (this probe)       {num_workers}")
    for w in (8, 12, 16, 20):
        _p(f"  => with num_workers={w:<3}         {t_prod/w:8.3f} s/step sustainable")
    _p("")
    _p("  measured from training logs, for comparison:")
    _p("    base 1n  T_prod 80.5 s  (1.678 s/sample)   C 8.875 s   e2e 10.066 s")
    _p("    2n ep3   T_prod 77.9 s  (1.624 s/sample)   C 7.922 s   e2e  9.742 s")
    _p("  A single-process probe has no worker contention, so expect this number")
    _p("  to land at or below the log-derived one. Interpret the STAGE SPLIT and")
    _p("  the gather A/B ratio, not the absolute floor.")
    _p("[supply] -----------------------------------------------------------")


def main():
    if len(sys.argv) < 2:
        _p("usage: python tools/probe_dataloader_cost.py <config.yaml>")
        sys.exit(1)

    threads = os.environ.get("PROBE_THREADS")
    if threads:
        torch.set_num_threads(int(threads))
    _p(f"[probe] torch.get_num_threads() = {torch.get_num_threads()}")
    _p(f"[probe] OMP_NUM_THREADS         = {os.environ.get('OMP_NUM_THREADS')}")

    n = int(os.environ.get("PROBE_N", 24))
    warmup = int(os.environ.get("PROBE_WARMUP", 3))
    seed = int(os.environ.get("PROBE_SEED", 1234))

    t0 = time.time()
    args, ds = build_dataset()
    _p(f"[probe] dataset built in {time.time()-t0:.1f} s, {len(ds)} samples")

    sub, lr = describe_layout(ds)
    probe_gather_variants(lr, n_trials=max(4, n // 2), seed=seed)
    per_sample = probe_stages(ds, sub, lr, n=n, warmup=warmup, seed=seed)
    report_supply_math(per_sample, args.train.micro_batch_size, args.data.num_workers)


if __name__ == "__main__":
    main()
