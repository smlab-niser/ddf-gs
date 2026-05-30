"""DDF Global-Illumination renderer — DDF as a secondary-ray oracle.

The headline application: a DDF distilled from a GS/implicit scene answers
secondary-ray visibility queries (shadows, AO, GI bounces) that GS rasterization
cannot produce, in ONE network eval per ray.

Design: **callback injection.** Each render tier takes oracle callbacks
(intersect / occlude / normal / color). Passing DDF callbacks gives DDF-GI;
passing embree-mesh callbacks gives the ground-truth reference with *identical*
shading math — so PSNR isolates the oracle, not the renderer.

Tier 1 (this file, first): hard/soft shadows + ambient occlusion. No color head
needed. Most robust to DDF normal noise (visibility, not orientation).
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Geometry-agnostic model access (handles DDFHashGrid 2-tuple AND
# DDFHashGridColor 3-tuple uniformly).
# ----------------------------------------------------------------------------

def _geom(model, x, d):
    """Return (dist, vis_logit) for any DDF variant."""
    if hasattr(model, "forward_geom"):
        return model.forward_geom(x, d)
    out = model(x, d)
    return out[0], out[1]


# ----------------------------------------------------------------------------
# DDF oracle callbacks
# ----------------------------------------------------------------------------

@torch.no_grad()
def ddf_sphere_trace(model, origins, dirs, *, eps=0.05, max_iter=48, t_far=5.0,
                     t_min=0.0, chunk=200000, bbox_half=1.2, vis_thresh=0.5):
    """Primary-hit finder. Returns (hit, x_surface, t, vis_logit).

    Hit requires: sphere-trace converged (residual<eps) AND vis>thresh AND the
    surface point is inside the bbox (kills false hits on background rays — the
    DDF reports spurious near-surface residuals in empty space otherwise).
    """
    N = origins.shape[0]
    device = origins.device
    hit = torch.zeros(N, dtype=torch.bool, device=device)
    x_out = origins.clone()
    t_out = torch.full((N,), t_min, device=device)
    vis_out = torch.full((N,), -10.0, device=device)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        o, d = origins[s:e], dirs[s:e]
        t = torch.full((o.shape[0],), t_min, device=device)
        alive = torch.ones(o.shape[0], dtype=torch.bool, device=device)
        xf = o + t.unsqueeze(-1) * d
        vf = torch.full((o.shape[0],), -10.0, device=device)
        conv = torch.zeros(o.shape[0], dtype=torch.bool, device=device)
        for _ in range(max_iter):
            if not alive.any():
                break
            idx = alive.nonzero(as_tuple=True)[0]
            x = o[idx] + t[idx].unsqueeze(-1) * d[idx]
            dist, vis = _geom(model, x, d[idx])
            vf[idx] = vis
            is_hit = dist < eps
            if is_hit.any():
                hi = idx[is_hit]
                xf[hi] = x[is_hit]
                conv[hi] = True
                alive[hi] = False
            t[idx] = t[idx] + dist.clamp(min=eps * 0.5, max=t_far)
            esc = t[idx] > t_far
            if esc.any():
                alive[idx[esc]] = False
        hit[s:e] = conv
        x_out[s:e] = xf
        t_out[s:e] = t
        vis_out[s:e] = vf
    in_bbox = (x_out.abs() < bbox_half).all(dim=-1)
    hit = hit & (torch.sigmoid(vis_out) > vis_thresh) & in_bbox
    return hit, x_out, t_out, vis_out


@torch.no_grad()
def ddf_occluded(model, origins, dirs, max_dist, *, vis_thresh=0.5,
                 t_self=0.02, chunk=500000):
    """ONE-EVAL visibility query: is there an occluder within max_dist?

    This is the headline O(1) operation — a single forward pass per ray,
    no iteration. occluded iff sigmoid(vis)>thresh AND t in (t_self, max_dist).
    """
    N = origins.shape[0]
    device = origins.device
    occ = torch.zeros(N, dtype=torch.bool, device=device)
    md = max_dist if torch.is_tensor(max_dist) else torch.full((N,), max_dist, device=device)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        dist, vis = _geom(model, origins[s:e], dirs[s:e])
        hit = torch.sigmoid(vis) > vis_thresh
        within = (dist > t_self) & (dist < md[s:e])
        occ[s:e] = hit & within
    return occ


def ddf_normals(model, x, dirs, *, chunk=65536):
    """Surface normals via autograd of distance w.r.t. position. Oriented later."""
    N = x.shape[0]
    device = x.device
    out = torch.zeros((N, 3), device=device)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        xi = x[s:e].detach().clone().requires_grad_(True)
        di = dirs[s:e]
        dist, _ = _geom(model, xi, di)
        grad = torch.autograd.grad(dist.sum(), xi, create_graph=False)[0]
        n = -grad
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out[s:e] = n
    return out


# ----------------------------------------------------------------------------
# Embree mesh oracle callbacks (ground-truth reference)
# ----------------------------------------------------------------------------

class MeshOracle:
    """Wraps a trimesh embree intersector for the reference renderer.

    Build from an already-normalized trimesh (use GTMeshSupervisor to load
    frame-correctly), so DDF and mesh share the exact same world frame.
    """

    def __init__(self, mesh, device="cuda"):
        import trimesh
        self.mesh = mesh
        self.device = device
        self.intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    def intersect(self, origins, dirs):
        """Returns (hit, x_surface, t, normals_geom) as torch tensors."""
        o = origins.detach().cpu().numpy()
        d = dirs.detach().cpu().numpy()
        locs, ray_idx, tri_idx = self.intersector.intersects_location(
            o, d, multiple_hits=False)
        N = o.shape[0]
        hit = torch.zeros(N, dtype=torch.bool, device=self.device)
        x = torch.from_numpy(o).float().to(self.device)
        nrm = torch.zeros((N, 3), device=self.device)
        if len(ray_idx) > 0:
            import numpy as np
            xl = torch.from_numpy(locs).float().to(self.device)
            fn = torch.from_numpy(self.mesh.face_normals[tri_idx]).float().to(self.device)
            ri = torch.from_numpy(ray_idx).long().to(self.device)
            hit[ri] = True
            x[ri] = xl
            nrm[ri] = fn
        t = (x - origins).norm(dim=-1)
        return hit, x, t, nrm

    def occluded(self, origins, dirs, max_dist, t_self=0.02):
        o = origins.detach().cpu().numpy()
        d = dirs.detach().cpu().numpy()
        locs, ray_idx, _ = self.intersector.intersects_location(
            o, d, multiple_hits=False)
        N = o.shape[0]
        occ = torch.zeros(N, dtype=torch.bool, device=self.device)
        if len(ray_idx) > 0:
            import numpy as np
            dist = np.linalg.norm(locs - o[ray_idx], axis=1)
            md = (max_dist.detach().cpu().numpy() if torch.is_tensor(max_dist)
                  else np.full(N, max_dist))
            keep = (dist > t_self) & (dist < md[ray_idx])
            ri = ray_idx[keep]
            occ_np = np.zeros(N, dtype=bool)
            occ_np[ri] = True
            occ = torch.from_numpy(occ_np).to(self.device)
        return occ


# ----------------------------------------------------------------------------
# Tier 1: shadows + AO  (oracle-agnostic via callbacks)
# ----------------------------------------------------------------------------

def orient_normals(n, view_dirs):
    """Flip normals to face the camera: n . (-view_dir) > 0."""
    flip = (n * view_dirs).sum(-1) > 0  # n points along ray => away from camera
    n = n.clone()
    n[flip] = -n[flip]
    return n


def render_shadows(
    intersect_fn, occlude_fn, normal_fn,
    origins, dirs, light_dir, *,
    albedo=(0.72, 0.72, 0.72), ambient=0.25,
    shadow_offset=0.08, bg=(1.0, 1.0, 1.0), device="cuda",
):
    """Tier-1 hard directional shadows. Oracle-agnostic.

    Returns dict(rgb (N,3), hit (N,), shadow (N,), normals (N,3), depth (N,)).
    light_dir: unit vector the light travels ALONG (so shade dir = -light_dir).
    """
    N = origins.shape[0]
    hit, x, t, nrm_geom = intersect_fn(origins, dirs)

    # Normals: use oracle's if provided (mesh), else compute (DDF)
    if normal_fn is not None:
        nrm = normal_fn(x, dirs)
    else:
        nrm = nrm_geom
    nrm = orient_normals(nrm, dirs)

    to_light = (-light_dir).expand(N, 3) if light_dir.dim() == 1 else -light_dir
    to_light = to_light / to_light.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # Shadow rays from offset surface points toward the light
    shadow = torch.zeros(N, dtype=torch.bool, device=device)
    if hit.any():
        hi = hit.nonzero(as_tuple=True)[0]
        sh_o = x[hi] + shadow_offset * nrm[hi]
        sh_d = to_light[hi]
        occ = occlude_fn(sh_o, sh_d, torch.full((hi.shape[0],), 10.0, device=device))
        shadow[hi] = occ

    # Lambertian shading
    ndotl = (nrm * to_light).sum(-1).clamp(min=0.0)
    alb = torch.tensor(albedo, device=device).view(1, 3)
    lit = ambient + (1.0 - ambient) * ndotl.unsqueeze(-1) * (~shadow).float().unsqueeze(-1)
    rgb = alb * lit
    bg_t = torch.tensor(bg, device=device).view(1, 3)
    rgb = torch.where(hit.unsqueeze(-1), rgb, bg_t)
    return {"rgb": rgb, "hit": hit, "shadow": shadow, "normals": nrm, "depth": t}


def render_ao(
    intersect_fn, occlude_fn, normal_fn,
    origins, dirs, *, n_ao=32, ao_radius=0.5, shadow_offset=0.08,
    albedo=(0.72, 0.72, 0.72), bg=(1.0, 1.0, 1.0), seed=0, device="cuda",
):
    """Tier-1 ambient occlusion. n_ao cosine-hemisphere rays per surface point."""
    N = origins.shape[0]
    hit, x, t, nrm_geom = intersect_fn(origins, dirs)
    nrm = normal_fn(x, dirs) if normal_fn is not None else nrm_geom
    nrm = orient_normals(nrm, dirs)

    ao = torch.ones(N, device=device)
    if hit.any():
        hi = hit.nonzero(as_tuple=True)[0]
        M = hi.shape[0]
        n_h = nrm[hi]
        # orthonormal basis per normal
        up = torch.where(n_h[:, 2:3].abs() < 0.9,
                         torch.tensor([0., 0., 1.], device=device).expand(M, 3),
                         torch.tensor([1., 0., 0.], device=device).expand(M, 3))
        tang = torch.cross(up, n_h, dim=-1); tang = tang / tang.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        bitan = torch.cross(n_h, tang, dim=-1)
        g = torch.Generator(device=device).manual_seed(seed)
        occ_frac = torch.zeros(M, device=device)
        sh_o = (x[hi] + shadow_offset * n_h)
        for _ in range(n_ao):
            u1 = torch.rand(M, device=device, generator=g)
            u2 = torch.rand(M, device=device, generator=g)
            r = u1.sqrt(); theta = 2 * math.pi * u2
            lx = r * torch.cos(theta); ly = r * torch.sin(theta)
            lz = (1 - u1).clamp(min=0).sqrt()
            d = lx.unsqueeze(-1) * tang + ly.unsqueeze(-1) * bitan + lz.unsqueeze(-1) * n_h
            d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            occ = occlude_fn(sh_o, d, torch.full((M,), ao_radius, device=device))
            occ_frac += occ.float()
        ao_hi = 1.0 - occ_frac / n_ao
        ao[hi] = ao_hi

    alb = torch.tensor(albedo, device=device).view(1, 3)
    rgb = alb * ao.unsqueeze(-1)
    bg_t = torch.tensor(bg, device=device).view(1, 3)
    rgb = torch.where(hit.unsqueeze(-1), rgb, bg_t)
    return {"rgb": rgb, "hit": hit, "ao": ao, "normals": nrm, "depth": t}
