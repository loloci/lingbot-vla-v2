from typing import *
import math
import os
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.types
import utils3d

from .tools import timeit
from .geometry_numpy import solve_optimal_focal_shift, solve_optimal_shift


# ---- Debug dump hook for offline scipy-vs-GPU validation ------------------
# When env var LINGBOT_FOCAL_SHIFT_DUMP_DIR is set, the first few calls to
# recover_focal_shift() will dump (points, mask, focal) to disk so an offline
# validation script can compare solvers on the exact tensors the trainer uses.
_FOCAL_SHIFT_DUMP_STATE: Dict[str, int] = {"dumped": 0, "max": 4}


def _maybe_dump_focal_shift_inputs(points: torch.Tensor,
                                   mask: Optional[torch.Tensor],
                                   focal: Optional[torch.Tensor]) -> None:
    dump_dir = os.environ.get("LINGBOT_FOCAL_SHIFT_DUMP_DIR")
    if not dump_dir:
        return
    if _FOCAL_SHIFT_DUMP_STATE["dumped"] >= _FOCAL_SHIFT_DUMP_STATE["max"]:
        return
    try:
        os.makedirs(dump_dir, exist_ok=True)
    except OSError:
        return
    idx = _FOCAL_SHIFT_DUMP_STATE["dumped"]
    path = os.path.join(dump_dir, f"focal_shift_call_{idx:02d}.pt")
    payload = {
        "points": points.detach().cpu(),
        "mask": mask.detach().cpu() if mask is not None else None,
        "focal": focal.detach().cpu() if focal is not None else None,
    }
    torch.save(payload, path)
    _FOCAL_SHIFT_DUMP_STATE["dumped"] += 1


def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)


def harmonic_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.add(eps).reciprocal().mean(dim=dim, keepdim=keepdim).reciprocal()
    else:
        w = w.to(x.dtype)
        return weighted_mean(x.add(eps).reciprocal(), w, dim=dim, keepdim=keepdim, eps=eps).add(eps).reciprocal()


def geometric_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.add(eps).log().mean(dim=dim).exp()
    else:
        w = w.to(x.dtype)
        return weighted_mean(x.add(eps).log(), w, dim=dim, keepdim=keepdim, eps=eps).exp()


def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def gaussian_blur_2d(input: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    kernel = torch.exp(-(torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=input.dtype, device=input.device) ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :]).reshape(1, 1, kernel_size, kernel_size)
    input = F.pad(input, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode='replicate')
    input = F.conv2d(input, kernel, groups=input.shape[1])
    return input


def focal_to_fov(focal: torch.Tensor):
    return 2 * torch.atan(0.5 / focal)


def fov_to_focal(fov: torch.Tensor):
    return 0.5 / torch.tan(fov / 2)


def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))

def intrinsics_to_fov(intrinsics: torch.Tensor):
    """
    Returns field of view in radians from normalized intrinsics matrix.
    ### Parameters:
    - intrinsics: torch.Tensor of shape (..., 3, 3)

    ### Returns:
    - fov_x: torch.Tensor of shape (...)
    - fov_y: torch.Tensor of shape (...)
    """
    focal_x = intrinsics[..., 0, 0]
    focal_y = intrinsics[..., 1, 1]
    return 2 * torch.atan(0.5 / focal_x), 2 * torch.atan(0.5 / focal_y)


def point_map_to_depth_legacy(points: torch.Tensor):
    height, width = points.shape[-3:-1]
    diagonal = (height ** 2 + width ** 2) ** 0.5
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)  # (H, W, 2)

    # Solve least squares problem
    b = (uv * points[..., 2:]).flatten(-3, -1)                        # (..., H * W * 2)
    A = torch.stack([points[..., :2], -uv.expand_as(points[..., :2])], dim=-1).flatten(-4, -2)   # (..., H * W * 2, 2)

    M = A.transpose(-2, -1) @ A 
    solution = (torch.inverse(M + 1e-6 * torch.eye(2).to(A)) @ (A.transpose(-2, -1) @ b[..., None])).squeeze(-1)
    focal, shift = solution.unbind(-1)

    depth = points[..., 2] + shift[..., None, None]
    fov_x = torch.atan(width / diagonal / focal) * 2
    fov_y = torch.atan(height / diagonal / focal) * 2
    return depth, fov_x, fov_y, shift


def view_plane_uv_to_focal(uv: torch.Tensor):
    normed_uv = normalized_view_plane_uv(width=uv.shape[-2], height=uv.shape[-3], device=uv.device, dtype=uv.dtype)
    focal = (uv * normed_uv).sum() / uv.square().sum().add(1e-12)
    return focal


# --- New: batched, on-device solver for recover_focal_shift ---------------
#
# Goal: eliminate the ~72 ms scipy L-M stall inside depth_teacher_forward by
# doing the same 2-parameter fit (focal, shift) entirely on GPU with a
# fixed-iteration Gauss-Newton / Levenberg-Marquardt loop. Both loop count and
# tensor shapes are static, so the whole function is torch.compile-friendly.
#
# Objective (matches solve_optimal_focal_shift in geometry_numpy.py):
#     minimise  sum_i w_i * || f * xy_i / (z_i + s) - uv_i ||^2
#
# Invalid pixels are folded in as weight 0 rather than through nonzero() so we
# keep static shapes. Per-image "<2 valid pixels" degenerate case returns
# (focal=1, shift=0) via a boolean fallback mask.


def _linear_focal_shift_gpu(uv: torch.Tensor,
                            xy: torch.Tensor,
                            z: torch.Tensor,
                            w: torch.Tensor,
                            eps: float = 1e-6,
                            damping: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear initial guess for (focal, shift).

    Rearranging ``f * xy / (z + s) = uv`` as ``f * xy - s * uv = uv * z`` gives a
    per-pixel linear equation in the two unknowns (f, s). Stack over all pixels
    (x and y components), weight by ``w``, and solve the 2x2 normal equations.

    Shapes:
    - ``uv``, ``xy``: ``[B, N, 2]``
    - ``z``, ``w``: ``[B, N]``
    Returns two tensors of shape ``[B]``.
    """
    B, N, _ = uv.shape
    # A_i,x = [ xy_x_i, -uv_x_i ],  b_i,x = uv_x_i * z_i
    # A_i,y = [ xy_y_i, -uv_y_i ],  b_i,y = uv_y_i * z_i
    A = torch.stack([xy, -uv], dim=-1)                # [B, N, 2, 2]  last dim = param
    A = A.reshape(B, N * 2, 2)                        # [B, 2N, 2]
    b = (uv * z.unsqueeze(-1)).reshape(B, N * 2)      # [B, 2N]
    w2 = w.unsqueeze(-1).expand(-1, -1, 2).reshape(B, N * 2)  # per-row weight

    Aw = A * w2.unsqueeze(-1)
    AtA = Aw.transpose(-2, -1) @ A                    # [B, 2, 2]
    Atb = (Aw.transpose(-2, -1) @ b.unsqueeze(-1)).squeeze(-1)  # [B, 2]

    I2 = torch.eye(2, device=A.device, dtype=A.dtype).expand(B, 2, 2)
    sol = torch.linalg.solve(AtA + damping * I2, Atb.unsqueeze(-1)).squeeze(-1)  # [B, 2]
    focal = sol[..., 0]
    shift = sol[..., 1]
    return focal, shift


def _refine_focal_shift_lm(uv: torch.Tensor,
                           xy: torch.Tensor,
                           z: torch.Tensor,
                           w: torch.Tensor,
                           focal: torch.Tensor,
                           shift: torch.Tensor,
                           num_iterations: int = 6,
                           damping: float = 1e-4,
                           denom_floor: float = 1e-4,
                           focal_floor: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed-step Gauss-Newton / Levenberg-Marquardt refinement on (focal, shift).

    Residuals per pixel (x and y components stacked):
        r = f * xy / (z + s) - uv

    Jacobian entries (per pixel, per component):
        d r / d f = xy / (z + s)
        d r / d s = - f * xy / (z + s)^2

    We assemble the [2, 2] normal system JᵀWJ + λI, solve for the update, and
    step. ``num_iterations`` is a Python int → the loop unrolls in torch.compile.
    Both ``focal`` and ``(z + shift)`` are floored to keep the projection well
    defined even for degenerate batches.
    """
    for _ in range(num_iterations):
        denom = (z + shift.unsqueeze(-1)).clamp_min(denom_floor)  # [B, N]
        inv_denom = 1.0 / denom
        xy_proj = xy * inv_denom.unsqueeze(-1)                    # [B, N, 2]

        residual = focal.unsqueeze(-1).unsqueeze(-1) * xy_proj - uv  # [B, N, 2]

        # J columns
        jf = xy_proj                                                # d r / d f
        js = -focal.unsqueeze(-1).unsqueeze(-1) * xy_proj * inv_denom.unsqueeze(-1)

        # Flatten (N, 2) into rows, apply weight
        r_flat = residual.reshape(residual.shape[0], -1)            # [B, 2N]
        jf_flat = jf.reshape(jf.shape[0], -1)
        js_flat = js.reshape(js.shape[0], -1)
        w_flat = w.unsqueeze(-1).expand(-1, -1, 2).reshape(w.shape[0], -1)

        # Weighted JᵀJ and Jᵀr (both 2x2 / 2-vector per batch)
        jff = (w_flat * jf_flat * jf_flat).sum(dim=-1)
        jfs = (w_flat * jf_flat * js_flat).sum(dim=-1)
        jss = (w_flat * js_flat * js_flat).sum(dim=-1)
        jtj = torch.stack([torch.stack([jff, jfs], dim=-1),
                           torch.stack([jfs, jss], dim=-1)], dim=-2)  # [B, 2, 2]

        jtr_f = (w_flat * jf_flat * r_flat).sum(dim=-1)
        jtr_s = (w_flat * js_flat * r_flat).sum(dim=-1)
        jtr = torch.stack([jtr_f, jtr_s], dim=-1)                     # [B, 2]

        I2 = torch.eye(2, device=jtj.device, dtype=jtj.dtype).expand_as(jtj)
        delta = torch.linalg.solve(jtj + damping * I2, jtr.unsqueeze(-1)).squeeze(-1)

        focal = focal - delta[..., 0]
        shift = shift - delta[..., 1]
        focal = focal.clamp_min(focal_floor)
    return focal, shift


def recover_focal_shift_gpu(points: torch.Tensor,
                            mask: torch.Tensor = None,
                            focal: torch.Tensor = None,
                            downsample_size: Tuple[int, int] = (64, 64),
                            num_iterations: int = 6,
                            damping: float = 1e-4,
                            solver: str = "gpu_lm") -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched GPU solver for (focal, shift). Mirrors ``recover_focal_shift``.

    Parameters
    ----------
    points : (..., H, W, 3) tensor.
    mask   : (..., H, W) bool/float tensor or None. Folded in as multiplicative
             weight; NEVER routed through ``nonzero()`` (dynamic shapes break
             ``torch.compile``).
    focal  : (...,) tensor or None. When given, only ``shift`` is optimised.
    downsample_size : (h_lr, w_lr). Same nearest downsample as the legacy path.
    num_iterations : fixed count of LM steps (used when ``solver='gpu_lm'``).
    damping : Levenberg-Marquardt damping added to JᵀJ diagonal.
    solver : ``'gpu_linear'`` — just the 2x2 linear solve.
             ``'gpu_lm'``    — linear init + ``num_iterations`` LM refinement.

    Returns
    -------
    focal, shift : (...,) tensors on the same device / dtype as ``points``.
    """
    if solver not in ("gpu_linear", "gpu_lm"):
        raise ValueError(f"recover_focal_shift_gpu: unknown solver {solver!r}")

    shape = points.shape
    height, width = shape[-3], shape[-2]

    points = points.reshape(-1, *shape[-3:])
    mask_r = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal_r = None if focal is None else focal.reshape(-1)

    B = points.shape[0]

    # UV grid (same convention as the legacy path).
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)  # [H, W, 2]

    # Nearest downsample to (h_lr, w_lr) on GPU.
    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode='nearest').permute(0, 2, 3, 1)  # [B, h, w, 3]
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)  # [h, w, 2]
    if mask_r is None:
        w_lr = torch.ones(B, downsample_size[0], downsample_size[1], device=points.device, dtype=points.dtype)
    else:
        # nearest interp on float, then binarise back — matches legacy behaviour
        w_lr = F.interpolate(mask_r.to(torch.float32).unsqueeze(1), downsample_size, mode='nearest').squeeze(1)
        w_lr = (w_lr > 0).to(points.dtype)

    # Force FP32 for the numerics — 2x2 normal equations are cheap and this
    # keeps the fit stable when the outer trainer runs in bf16 autocast.
    dtype_out = points.dtype
    points_f = points_lr.to(torch.float32)
    uv_f = uv_lr.to(torch.float32)
    w_f = w_lr.to(torch.float32)

    N = downsample_size[0] * downsample_size[1]
    xy = points_f[..., :2].reshape(B, N, 2)   # [B, N, 2]
    z = points_f[..., 2].reshape(B, N)        # [B, N]
    uvB = uv_f.reshape(N, 2).unsqueeze(0).expand(B, -1, -1)  # [B, N, 2]
    w_bn = w_f.reshape(B, N)                  # [B, N]

    valid_count = w_bn.sum(dim=-1)            # [B]

    if focal_r is None:
        # Linear init in 2 unknowns.
        f0, s0 = _linear_focal_shift_gpu(uvB, xy, z, w_bn, damping=damping)
        f0 = f0.clamp_min(1e-4)

        if solver == "gpu_linear":
            f_out, s_out = f0, s0
        else:  # gpu_lm
            f_out, s_out = _refine_focal_shift_lm(
                uvB, xy, z, w_bn, f0, s0,
                num_iterations=num_iterations,
                damping=damping,
            )
    else:
        # Focal is known → only refine shift.
        f0 = focal_r.to(torch.float32)
        # closed-form linear init for shift alone: (f * xy - uv * z) = uv * s
        # -> s = sum(w * uv . (f * xy - uv * z)) / sum(w * ||uv||^2)
        rhs = (uvB * (f0.unsqueeze(-1).unsqueeze(-1) * xy - uvB * z.unsqueeze(-1))).sum(dim=-1)  # [B, N]
        num = (w_bn * rhs).sum(dim=-1)
        den = (w_bn * (uvB * uvB).sum(dim=-1)).sum(dim=-1).clamp_min(1e-8)
        s0 = num / den

        if solver == "gpu_linear":
            f_out, s_out = f0, s0
        else:
            # Refine both, but pin focal at the caller-provided value each step.
            f_ref, s_out = _refine_focal_shift_lm(
                uvB, xy, z, w_bn, f0, s0,
                num_iterations=num_iterations,
                damping=damping,
            )
            f_out = f0  # discard focal updates; caller wants that fixed

    # Degenerate fallback: <2 valid pixels ⇒ (focal=1, shift=0). Done via where
    # so the shape stays static and there is no host sync.
    fallback = valid_count < 2
    f_out = torch.where(fallback, torch.ones_like(f_out), f_out)
    s_out = torch.where(fallback, torch.zeros_like(s_out), s_out)

    # NaN / Inf guard — if the 2x2 solve blew up on some batch entry we would
    # otherwise poison downstream ops. Fall back to the safe default there too.
    bad = ~torch.isfinite(f_out) | ~torch.isfinite(s_out)
    if bad.any():
        f_out = torch.where(bad, torch.ones_like(f_out), f_out)
        s_out = torch.where(bad, torch.zeros_like(s_out), s_out)

    f_out = f_out.to(dtype_out).reshape(shape[:-3])
    s_out = s_out.to(dtype_out).reshape(shape[:-3])
    return f_out, s_out


def recover_focal_shift(points: torch.Tensor,
                        mask: torch.Tensor = None,
                        focal: torch.Tensor = None,
                        downsample_size: Tuple[int, int] = (64, 64),
                        solver: str = "scipy",
                        num_iterations: int = 6,
                        damping: float = 1e-4):
    """Dispatch to CPU scipy L-M or the batched GPU solver.

    ``solver='scipy'``    — legacy path (per-image ``scipy.optimize.least_squares``).
    ``solver='gpu_linear'`` — batched linear least-squares only.
    ``solver='gpu_lm'``    — batched linear init + ``num_iterations`` LM refinement.

    Shape and return contract match the legacy function.
    """
    _maybe_dump_focal_shift_inputs(points, mask, focal)

    if solver == "scipy":
        return _recover_focal_shift_scipy(points, mask=mask, focal=focal, downsample_size=downsample_size)
    if solver in ("gpu_linear", "gpu_lm"):
        return recover_focal_shift_gpu(
            points, mask=mask, focal=focal,
            downsample_size=downsample_size,
            num_iterations=num_iterations,
            damping=damping,
            solver=solver,
        )
    raise ValueError(f"recover_focal_shift: unknown solver {solver!r} "
                     f"(expected one of 'scipy', 'gpu_linear', 'gpu_lm')")


def _recover_focal_shift_scipy(points: torch.Tensor, mask: torch.Tensor = None, focal: torch.Tensor = None, downsample_size: Tuple[int, int] = (64, 64)):
    """Legacy CPU L-M implementation (scipy.optimize.least_squares per image).

    Kept as validation baseline; new callers should prefer ``recover_focal_shift_gpu``
    via ``recover_focal_shift(..., solver='gpu_lm')`` for on-device execution.

    Recover the depth map and FoV from a point map with unknown z shift and focal.

    Note that it assumes:
    - the optical center is at the center of the map
    - the map is undistorted
    - the map is isometric in the x and y directions

    ### Parameters:
    - `points: torch.Tensor` of shape (..., H, W, 3)
    - `downsample_size: Tuple[int, int]` in (height, width), the size of the downsampled map. Downsampling produces approximate solution and is efficient for large maps.

    ### Returns:
    - `focal`: torch.Tensor of shape (...) the estimated focal length, relative to the half diagonal of the map
    - `shift`: torch.Tensor of shape (...) Z-axis shift to translate the point map to camera space
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]
    diagonal = (height ** 2 + width ** 2) ** 0.5

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)  # (H, W, 2)

    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode='nearest').permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)
    mask_lr = None if mask is None else F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode='nearest').squeeze(1) > 0
    
    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()
    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        points_lr_i_np = points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]
        if uv_lr_i_np.shape[0] < 2:
            optim_focal.append(1)
            optim_shift.append(0)
            continue
        if focal is None:
            optim_shift_i, optim_focal_i = solve_optimal_focal_shift(uv_lr_i_np, points_lr_i_np)
            optim_focal.append(float(optim_focal_i))
        else:
            optim_shift_i = solve_optimal_shift(uv_lr_i_np, points_lr_i_np, focal_np[i])
        optim_shift.append(float(optim_shift_i))
    optim_shift = torch.tensor(optim_shift, device=points.device, dtype=points.dtype).reshape(shape[:-3])

    if focal is None:
        optim_focal = torch.tensor(optim_focal, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        optim_focal = focal.reshape(shape[:-3])

    return optim_focal, optim_shift


def theshold_depth_change(depth: torch.Tensor, mask: torch.Tensor, pooler: Literal['min', 'max'], rtol: float = 0.2, kernel_size: int = 3):
    *batch_shape, height, width = depth.shape
    depth = depth.reshape(-1, 1, height, width)
    mask = mask.reshape(-1, 1, height, width)
    if pooler =='max':
        pooled_depth = F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2)
        output_mask = pooled_depth > depth * (1 + rtol)
    elif pooler =='min':
        pooled_depth = -F.max_pool2d(-torch.where(mask, depth, torch.inf), kernel_size, stride=1, padding=kernel_size // 2)
        output_mask =  pooled_depth < depth * (1 - rtol)
    else:
        raise ValueError(f'Unsupported pooler: {pooler}')
    output_mask = output_mask.reshape(*batch_shape, height, width)
    return output_mask


def dilate_with_mask(input: torch.Tensor, mask: torch.BoolTensor, filter: Literal['min', 'max', 'mean', 'median'] = 'mean', iterations: int = 1) -> torch.Tensor:
    kernel = torch.tensor([[False, True, False], [True, True, True], [False, True, False]], device=input.device, dtype=torch.bool)
    for _ in range(iterations):
        input_window = utils3d.pt.sliding_window(F.pad(input, (1, 1, 1, 1), mode='constant', value=0), window_size=3, stride=1, dim=(-2, -1))
        mask_window = kernel & utils3d.pt.sliding_window(F.pad(mask, (1, 1, 1, 1), mode='constant', value=False), window_size=3, stride=1, dim=(-2, -1))    
        if filter =='min':
            input = torch.where(mask, input, torch.where(mask_window, input_window, torch.inf).min(dim=(-2, -1)).values)
        elif filter =='max':
            input = torch.where(mask, input, torch.where(mask_window, input_window, -torch.inf).max(dim=(-2, -1)).values)
        elif filter == 'mean':
            input = torch.where(mask, input, torch.where(mask_window, input_window, torch.nan).nanmean(dim=(-2, -1)))
        elif filter =='median':
            input = torch.where(mask, input, torch.where(mask_window, input_window, torch.nan).flatten(-2).nanmedian(dim=-1).values)
        mask = mask_window.any(dim=(-2, -1))
    return input, mask


def refine_depth_with_normal(depth: torch.Tensor, normal: torch.Tensor, intrinsics: torch.Tensor, iterations: int = 10, damp: float = 1e-3, eps: float = 1e-12, kernel_size: int = 5) -> torch.Tensor:
    device, dtype = depth.device, depth.dtype
    height, width = depth.shape[-2:]
    radius = kernel_size // 2

    duv = torch.stack(torch.meshgrid(torch.linspace(-radius / width, radius / width, kernel_size, device=device, dtype=dtype), torch.linspace(-radius / height, radius / height, kernel_size, device=device, dtype=dtype), indexing='xy'), dim=-1).to(dtype=dtype, device=device)

    log_depth = depth.clamp_min_(eps).log()
    log_depth_diff = utils3d.pt.sliding_window(log_depth, window_size=kernel_size, stride=1, dim=(-2, -1)) - log_depth[..., radius:-radius, radius:-radius, None, None] 
    
    weight = torch.exp(-(log_depth_diff / duv.norm(dim=-1).clamp_min_(eps) / 10).square())
    tot_weight = weight.sum(dim=(-2, -1)).clamp_min_(eps)

    uv = utils3d.pt.uv_map((height, width), device=device, dtype=dtype)
    K_inv = torch.inverse(intrinsics)

    grad = -(normal[..., None, :2] @ K_inv[..., None, None, :2, :2]).squeeze(-2) \
            / (normal[..., None, 2:] + normal[..., None, :2] @ (K_inv[..., None, None, :2, :2] @ uv[..., :, None] + K_inv[..., None, None, :2, 2:])).squeeze(-2)
    laplacian = (weight * ((utils3d.pt.sliding_window(grad, window_size=kernel_size, stride=1, dim=(-3, -2)) + grad[..., radius:-radius, radius:-radius, :, None, None]) * (duv.permute(2, 0, 1) / 2)).sum(dim=-3)).sum(dim=(-2, -1))
    
    laplacian = laplacian.clamp(-0.1, 0.1)
    log_depth_refine = log_depth.clone()

    for _ in range(iterations):
        log_depth_refine[..., radius:-radius, radius:-radius] = 0.1 * log_depth_refine[..., radius:-radius, radius:-radius] + 0.9 * (damp * log_depth[..., radius:-radius, radius:-radius] - laplacian + (weight * utils3d.pt.sliding_window_2d(log_depth_refine, window_size=kernel_size, stride=1, dim=(-2, -1))).sum(dim=(-2, -1))) / (tot_weight + damp) 

    depth_refine = log_depth_refine.exp()

    return depth_refine