"""Offline validation of GPU vs SciPy solvers for recover_focal_shift.

Runs the three solver variants on dumped (points, mask, focal) tensors from
LINGBOT_FOCAL_SHIFT_DUMP_DIR and reports per-call error stats vs the SciPy
reference (which is treated as ground truth).

Usage:
  # Step 1 (in trainer, one time): dump real inputs
  LINGBOT_FOCAL_SHIFT_DUMP_DIR=/tmp/focal_shift_dump  bash node1-run.sh ...

  # Step 2: validate
  python scripts/validate_focal_shift_gpu.py \
      --dump-dir /tmp/focal_shift_dump \
      --device cuda \
      --iters 3 6 10
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch

# --- utils3d stub -----------------------------------------------------------
# geometry_torch imports utils3d at module load. The validator does not use any
# utils3d functionality (only recover_focal_shift* which is self-contained), so
# a lightweight ModuleType stub lets us run the script in any env where MoGe's
# geometry helpers are available.
import types as _types
for _mod in ("utils3d", "utils3d.pt", "utils3d.np"):
    if _mod not in sys.modules:
        sys.modules[_mod] = _types.ModuleType(_mod)

# Add MoGe to path if not already importable.
_MOGE_ROOT = Path(__file__).resolve().parents[1] / \
    "lingbotvla/models/vla/vision_models/MoGe"
if _MOGE_ROOT.exists():
    sys.path.insert(0, str(_MOGE_ROOT))

from moge.utils.geometry_torch import recover_focal_shift  # noqa: E402


def summarise(name: str, ref_f: torch.Tensor, ref_s: torch.Tensor,
              cand_f: torch.Tensor, cand_s: torch.Tensor,
              depth_ref: torch.Tensor, mask_ref: torch.Tensor) -> None:
    """Print per-batch max/p50/p95 focal rel err, shift abs err, depth rel err."""
    focal_rel = (cand_f - ref_f).abs() / ref_f.abs().clamp_min(1e-8)
    shift_abs = (cand_s - ref_s).abs()

    # depth: shift adds directly to z, so per-pixel relative error is
    #   |z + s_c - (z + s_r)| / |z + s_r| = |s_c - s_r| / |z + s_r|
    # summarised over valid pixels only.
    z = depth_ref[..., 2]  # (B,H,W)
    ref_depth = z + ref_s.view(-1, 1, 1)
    cand_depth = z + cand_s.view(-1, 1, 1)
    denom = ref_depth.abs().clamp_min(1e-8)
    per_pixel = (cand_depth - ref_depth).abs() / denom  # (B,H,W)
    if mask_ref is not None:
        m = mask_ref
        valid = per_pixel[m]
    else:
        valid = per_pixel.reshape(-1)
    if valid.numel() == 0:
        p50 = p95 = mx = float("nan")
    else:
        vsorted, _ = valid.sort()
        n = vsorted.numel()
        p50 = vsorted[n // 2].item()
        p95 = vsorted[int(n * 0.95)].item() if n > 20 else vsorted[-1].item()
        mx = vsorted[-1].item()

    nan_f = int(torch.isnan(cand_f).sum().item()) + int(torch.isinf(cand_f).sum().item())
    nan_s = int(torch.isnan(cand_s).sum().item()) + int(torch.isinf(cand_s).sum().item())

    print(f"  {name:16s}  focal_rel  max={focal_rel.max().item():.2e}  "
          f"mean={focal_rel.mean().item():.2e}   "
          f"shift_abs  max={shift_abs.max().item():.2e}  mean={shift_abs.mean().item():.2e}")
    print(f"  {'':16s}  depth_rel  p50={p50:.2e}  p95={p95:.2e}  max={mx:.2e}  "
          f"NaN/Inf(f/s)={nan_f}/{nan_s}")


def timed(name: str, fn) -> tuple:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    print(f"    [{name}: {dt:.2f} ms]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True, help="LINGBOT_FOCAL_SHIFT_DUMP_DIR contents")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", nargs="*", type=int, default=[3, 6, 10],
                    help="LM iteration counts to compare (default 3 6 10)")
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32",
                    help="dtype at which to invoke recover_focal_shift (matches trainer autocast)")
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    files = sorted(dump_dir.glob("focal_shift_call_*.pt"))
    if not files:
        raise FileNotFoundError(f"no dumps in {dump_dir}")

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    for path in files:
        blob = torch.load(path, map_location="cpu")
        points = blob["points"].to(device=args.device, dtype=dtype)
        mask = blob["mask"].to(device=args.device) if blob["mask"] is not None else None
        focal_in = blob["focal"]
        if focal_in is not None:
            focal_in = focal_in.to(device=args.device, dtype=dtype)

        B = points.shape[0] if points.ndim == 4 else 1
        H, W = points.shape[-3], points.shape[-2]
        print(f"\n=== {path.name} : points {tuple(points.shape)} dtype={points.dtype} "
              f"mask={None if mask is None else tuple(mask.shape)} "
              f"focal_in={'given' if focal_in is not None else 'none'} ===")

        # SciPy reference (runs on CPU internally regardless of device).
        f_sci, s_sci = timed(
            "scipy",
            lambda: recover_focal_shift(points, mask=mask, focal=focal_in, solver="scipy"),
        )
        print(f"  scipy   focal={f_sci.detach().float().tolist()}  shift={s_sci.detach().float().tolist()}")

        f_lin, s_lin = timed(
            "gpu_linear",
            lambda: recover_focal_shift(points, mask=mask, focal=focal_in, solver="gpu_linear"),
        )
        summarise("gpu_linear",
                  f_sci.float(), s_sci.float(),
                  f_lin.float(), s_lin.float(),
                  points.float(), mask)

        for n in args.iters:
            f_lm, s_lm = timed(
                f"gpu_lm({n})",
                lambda n=n: recover_focal_shift(
                    points, mask=mask, focal=focal_in,
                    solver="gpu_lm", num_iterations=n,
                ),
            )
            summarise(f"gpu_lm(N={n})",
                      f_sci.float(), s_sci.float(),
                      f_lm.float(), s_lm.float(),
                      points.float(), mask)


if __name__ == "__main__":
    main()
