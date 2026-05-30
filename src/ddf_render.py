"""Direct per-ray DDF rendering (Stage 12).

Skip the UDF -> marching cubes pipeline that smooths out fine geometry
(see Stage 11 verdict). Instead:

  for each pixel ray (o, d):       # o = camera center, d = unit world-dir
      t, vis_logit = DDF(o, d)
      p = o + t * d                # surface point
      alpha = sigmoid(vis_logit)   # visibility = ray hits the surface
      n  = -normalize( grad_x DDF.distance(x, d) at x = p )
      shade Lambertian with one head-on light + ambient

This validates the DDF as a *queryable* representation (the paper's actual
contribution, following Behera & Mishra 2023, arXiv:2306.16142). No mesh
extraction, no marching cubes.
"""

from __future__ import annotations

import torch

from .gs_supervisor import _pixels_to_world_rays, _spherical_camera


def _normal_at_surface(
    model,
    surface_pts: torch.Tensor,
    dirs: torch.Tensor,
    chunk: int = 65536,
) -> torch.Tensor:
    """Estimate surface normals via grad_x DDF.distance(x, d) at x = surface point.

    n  = -normalize(grad). We negate because DDF.distance grows as x moves
    away from the surface along the ray, so its gradient points *into* the
    surface; the outward normal is the opposite direction.

    Done in chunks to stay within memory on big images.
    """
    device = surface_pts.device
    n_pts = surface_pts.shape[0]
    out = torch.zeros((n_pts, 3), device=device, dtype=torch.float32)
    for s in range(0, n_pts, chunk):
        e = min(s + chunk, n_pts)
        x = surface_pts[s:e].detach().clone().requires_grad_(True)
        d = dirs[s:e]
        dist, _ = model(x, d)
        grad = torch.autograd.grad(
            outputs=dist.sum(),
            inputs=x,
            create_graph=False,
            retain_graph=False,
        )[0]
        n = -grad
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out[s:e] = n
    return out


def _forward_in_chunks(
    model,
    origins: torch.Tensor,
    dirs: torch.Tensor,
    chunk: int = 262144,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = origins.shape[0]
    ts = torch.empty(n, device=origins.device, dtype=torch.float32)
    vs = torch.empty(n, device=origins.device, dtype=torch.float32)
    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            t, v = model(origins[s:e], dirs[s:e])
            ts[s:e] = t
            vs[s:e] = v
    return ts, vs


def render_ddf(
    model,
    w2c: torch.Tensor,        # unused but kept for API symmetry / future use
    K: torch.Tensor,
    c2w: torch.Tensor,
    image_size: int,
    device: str = "cuda",
    light_dir_cam: tuple[float, float, float] = (0.3, -0.4, -1.0),
    ambient: float = 0.30,
    base_color: tuple[float, float, float] = (0.72, 0.78, 0.88),
    vis_thresh: float = 0.5,
    fwd_chunk: int = 262144,
    grad_chunk: int = 65536,
) -> dict:
    """Direct DDF render.

    Args:
        model: DDF (DDFHashGrid or DDF) -- model(points, dirs) -> (t, vis_logit).
        w2c, K, c2w: standard OpenCV camera matrices (3x3 K, 4x4 w2c/c2w).
        image_size: square HxW.
        device: cuda string.
        light_dir_cam: directional light in camera space; will be transformed
            to world. Z-forward (OpenCV); -z is "toward camera".
        ambient: scalar baseline brightness for non-lit surfaces.
        base_color: Lambertian albedo (R, G, B).
        vis_thresh: visibility sigmoid threshold to call a pixel a "hit".
        fwd_chunk: ray chunk for forward DDF pass (no grad).
        grad_chunk: ray chunk for normal estimation (autograd).

    Returns dict with:
        rgb     (H, W, 3) uint8 -- composited on white background
        depth   (H, W)    float32 -- t along the ray (NaN where !alpha)
        alpha   (H, W)    bool    -- vis_logit.sigmoid() > vis_thresh
        normals (H, W, 3) float32 -- world-space outward normal (0 where !alpha)
    """
    model.eval()
    H = W = image_size

    origins_grid, dirs_grid = _pixels_to_world_rays(K, c2w, image_size, device=device)
    origins = origins_grid.reshape(-1, 3).contiguous()
    dirs = dirs_grid.reshape(-1, 3).contiguous()

    t, vis_logit = _forward_in_chunks(model, origins, dirs, chunk=fwd_chunk)
    alpha = vis_logit.sigmoid() > vis_thresh

    # Surface points (only meaningful where alpha True; compute everywhere for
    # cheap, mask after).
    surf = origins + t.unsqueeze(-1) * dirs        # (N, 3)

    # Cap t at the bbox to keep gradients well-behaved if model extrapolates.
    bbox_half = getattr(model, "bbox_half", 1.2)
    surf = surf.clamp(-bbox_half * 1.05, bbox_half * 1.05)

    # Normals (autograd through DDF.distance head).
    alpha_idx = alpha.nonzero(as_tuple=False).squeeze(-1)
    normals_flat = torch.zeros_like(surf)
    if alpha_idx.numel() > 0:
        n_hit = _normal_at_surface(
            model,
            surf[alpha_idx],
            dirs[alpha_idx],
            chunk=grad_chunk,
        )
        normals_flat[alpha_idx] = n_hit

    # Shading. Lambertian + ambient. Light in world space: rotate light_dir_cam
    # by c2w[:3, :3].
    R = c2w[:3, :3]
    l_cam = torch.tensor(light_dir_cam, dtype=torch.float32, device=device)
    l_cam = l_cam / l_cam.norm().clamp_min(1e-8)
    # Light vector points *from surface toward light source*: that's -l_dir.
    l_world = R @ (-l_cam)
    l_world = l_world / l_world.norm().clamp_min(1e-8)

    ndotl = (normals_flat * l_world.view(1, 3)).sum(dim=-1).clamp(min=0.0)
    shade = (ambient + (1.0 - ambient) * ndotl).clamp(0.0, 1.0)  # (N,)

    color = torch.tensor(base_color, dtype=torch.float32, device=device).view(1, 3)
    rgb_lin = shade.unsqueeze(-1) * color                          # (N, 3) in [0,1]

    # Composite on white where !alpha.
    rgb_lin = torch.where(alpha.unsqueeze(-1), rgb_lin, torch.ones_like(rgb_lin))
    rgb_u8 = (rgb_lin.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

    rgb_img = rgb_u8.reshape(H, W, 3).contiguous()
    depth_img = torch.where(alpha, t, torch.full_like(t, float("nan"))).reshape(H, W)
    alpha_img = alpha.reshape(H, W)
    normals_img = normals_flat.reshape(H, W, 3)

    return {
        "rgb": rgb_img,
        "depth": depth_img,
        "alpha": alpha_img,
        "normals": normals_img,
    }


def render_ddf_spherical(
    model,
    elev_deg: float,
    azim_deg: float,
    radius: float,
    image_size: int,
    fov_deg: float = 60.0,
    device: str = "cuda",
    **kwargs,
) -> dict:
    """Convenience: build OpenCV spherical cam, then render."""
    w2c, K, c2w = _spherical_camera(
        elev_deg, azim_deg, radius, image_size, fov_deg=fov_deg, device=device,
    )
    return render_ddf(model, w2c, K, c2w, image_size, device=device, **kwargs)
