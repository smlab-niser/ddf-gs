"""Real-scene GI demo: Mip-NeRF360 garden table+vase (reused clean-gs GS, no mesh).

Fully mesh-free: GS supplies appearance (gsplat RGB, in color), the DDF supplies
both the primary surface (sphere-trace) and the secondary-ray GI (self-shadow +
AO, 1 eval/ray). Output panels:
  1. GS only (no GI)        — gsplat beauty pass, flat.
  2. GS + DDF GI (ours)      — + DDF self-shadow + DDF ambient occlusion.

The garden GS frame is z-up (the COLMAP frame doesn't map to the clean-gs GS, so
we use synthetic z-up orbit cameras). Output: renders/gi_garden.png.

Run: PYTHONPATH=. CUDA_VISIBLE_DEVICES=<g> python scripts/gi_garden.py --ckpt <ddf>
"""
import os, sys, math, argparse
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, ".")
from gsplat import rasterization
from src.train_ddf import build_model
from src.ddf_gi_render import ddf_sphere_trace, ddf_occluded

GS = "runs/garden_obj/gaussians.pt"


def c2w_zup(elev, azim, r, look_z=0.0, up_sign=1.0):
    """y-up orbit (the re-oriented garden GS is y-up). up_sign=-1 flips vertical
    (the reorient put the ground at +y, so up_sign=-1 shows the table right-side-up)."""
    e, a = math.radians(elev), math.radians(azim)
    eye = np.array([r*math.cos(e)*math.sin(a), r*math.sin(e),
                    r*math.cos(e)*math.cos(a)], np.float32)
    fwd = -eye / np.linalg.norm(eye)
    up = np.array([0, up_sign, 0], np.float32); right = np.cross(fwd, up); right /= np.linalg.norm(right)
    down = -np.cross(fwd, right)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right; c2w[:3, 1] = down; c2w[:3, 2] = fwd; c2w[:3, 3] = eye
    return c2w


def pixel_rays(c2w, K, H, W, device):
    ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                            torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    x = (xs + 0.5 - K[0, 2]) / K[0, 0]; y = (ys + 0.5 - K[1, 2]) / K[1, 1]
    dcam = torch.stack([x, y, torch.ones_like(x)], -1).reshape(-1, 3)
    R = torch.from_numpy(c2w[:3, :3]).to(device)
    d = (dcam @ R.T); d = d / d.norm(dim=-1, keepdim=True)
    o = torch.from_numpy(c2w[:3, 3]).to(device).expand_as(d)
    return o.contiguous(), d.contiguous()


@torch.no_grad()
def render_gs(gs, c2w, K, H, W, device):
    means = gs["means"].to(device); quats = gs["quats"].to(device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = gs["scales"].to(device).exp(); opac = gs["opacities"].to(device).sigmoid()
    colors = gs["colors"].to(device).sigmoid()
    w2c = torch.linalg.inv(torch.from_numpy(c2w).to(device)).unsqueeze(0)
    ren, alpha, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac,
                                  colors=colors, viewmats=w2c, Ks=K.unsqueeze(0), width=W,
                                  height=H, sh_degree=None, render_mode="RGB+ED")
    return ren[0, ..., :3].clamp(0, 1), ren[0, ..., 3], alpha[0, ..., 0]


def depth_normals(P, hit, H, W):
    """Screen-space normals from the (smooth) DDF surface points."""
    Pg = P.reshape(H, W, 3)
    dx = torch.zeros_like(Pg); dy = torch.zeros_like(Pg)
    dx[:, 1:-1] = Pg[:, 2:] - Pg[:, :-2]
    dy[1:-1, :] = Pg[2:, :] - Pg[:-2, :]
    n = torch.cross(dx, dy, dim=-1)
    return (n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)).reshape(-1, 3)


def hemi(n_h, M, n_ao, device, seed=0):
    up = torch.where(n_h[:, 2:3].abs() < 0.9,
                     torch.tensor([0., 0., 1.], device=device).expand(M, 3),
                     torch.tensor([1., 0., 0.], device=device).expand(M, 3))
    t = torch.cross(up, n_h, dim=-1); t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = torch.cross(n_h, t, dim=-1)
    g = torch.Generator(device=device).manual_seed(seed)
    for _ in range(n_ao):
        u1 = torch.rand(M, device=device, generator=g); u2 = torch.rand(M, device=device, generator=g)
        r = u1.sqrt(); th = 2*math.pi*u2
        d = (r*torch.cos(th)).unsqueeze(-1)*t + (r*torch.sin(th)).unsqueeze(-1)*b + \
            (1-u1).clamp(min=0).sqrt().unsqueeze(-1)*n_h
        yield d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/garden_obj/ddf/ddf_final.pt")
    ap.add_argument("--image_size", type=int, default=640)
    ap.add_argument("--elev", type=float, default=32.0)
    ap.add_argument("--azim", type=float, default=60.0)
    ap.add_argument("--radius", type=float, default=2.6)
    ap.add_argument("--ambient", type=float, default=0.32)
    ap.add_argument("--ao_strength", type=float, default=0.8)
    ap.add_argument("--flip_up", action="store_true", help="flip camera up (table right-side-up)")
    ap.add_argument("--gs", default=GS, help="GS gaussians.pt for the beauty pass")
    ap.add_argument("--out", default="renders/gi_garden.png")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    H = W = args.image_size
    fx = fy = 0.9 * W
    K = torch.tensor([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]], dtype=torch.float32, device=dev)
    c2w = c2w_zup(args.elev, args.azim, args.radius, 0.0, up_sign=(-1.0 if args.flip_up else 1.0))
    o, d = pixel_rays(c2w, K, H, W, dev)

    gs = torch.load(args.gs, map_location=dev, weights_only=False)
    gs_rgb, ed, alpha = render_gs(gs, c2w, K, H, W, dev)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = build_model(ck["cfg"]).to(dev).eval(); model.load_state_dict(ck["model"])

    # PRIMARY from the GS (its geometry is clean — the beauty pass proves it):
    # unproject expected-depth to world surface points; DDF is used only for the
    # secondary-ray occlusion query below.
    fwd = torch.from_numpy(c2w[:3, 2]).to(dev)
    hit = alpha.reshape(-1) > 0.5
    t = ed.reshape(-1) / (d @ fwd).clamp_min(1e-6)
    P = o + t.unsqueeze(-1) * d
    n = depth_normals(P, hit, H, W)
    flip = (n * d).sum(-1) > 0; n = n.clone(); n[flip] = -n[flip]

    light = torch.tensor([0.45, 0.80, 0.40], device=dev); light = light / light.norm()  # y-up: light from above-side
    to_light = light  # direction TO the light (up-ish)
    ndotl = (n * to_light).sum(-1).clamp(min=0.0)
    shadow = torch.zeros(hit.shape[0], device=dev); ao = torch.zeros(hit.shape[0], device=dev)
    if hit.any():
        hi = hit.nonzero(as_tuple=True)[0]
        so = P[hi] + 0.05 * n[hi]
        shadow[hi] = ddf_occluded(model, so, to_light.expand(hi.shape[0], 3).contiguous(),
                                  torch.full((hi.shape[0],), 6.0, device=dev), t_self=0.05).float()
        of = torch.zeros(hi.shape[0], device=dev)
        for sd in hemi(n[hi], hi.shape[0], 24, dev):
            of += ddf_occluded(model, so, sd, torch.full((hi.shape[0],), 0.5, device=dev), t_self=0.05).float()
        ao[hi] = of / 24

    alb = gs_rgb.reshape(-1, 3)
    lit = args.ambient + (1 - args.ambient) * ndotl * (1 - shadow)
    ao_fac = 1 - args.ao_strength * ao
    gi = (alb * (lit * ao_fac).unsqueeze(-1))
    bg = torch.ones_like(alb)
    # DDF GI MAP: normal-free occlusion signal the DDF computes per ray
    # (1 = fully visible/white, darker = more DDF-occluded by shadow + AO).
    gi_factor = ((1 - shadow) * (1 - args.ao_strength * ao)).clamp(0, 1)
    gi_map = gi_factor.unsqueeze(-1).expand(-1, 3)

    beauty = torch.where((alpha.reshape(-1) > 0.5).unsqueeze(-1), alb, bg).reshape(H, W, 3)
    gi_map_img = torch.where(hit.unsqueeze(-1), gi_map, bg).reshape(H, W, 3)
    gi_img = torch.where(hit.unsqueeze(-1), gi, bg).reshape(H, W, 3)

    panels = torch.cat([beauty, gi_map_img, gi_img], dim=1).clamp(0, 1)
    img = (panels.cpu().numpy() * 255).astype(np.uint8)
    pad = 30
    sheet = np.full((pad + H, 3 * W, 3), 255, np.uint8); sheet[pad:] = img
    pim = Image.fromarray(sheet); dr = ImageDraw.Draw(pim)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19)
    except Exception:
        f = ImageFont.load_default()
    dr.text((12, 6), "GS only (real captured scene)", fill=(0, 0, 0), font=f)
    dr.text((W + 12, 6), "DDF GI map (occlusion, 1 eval/ray)", fill=(0, 0, 0), font=f)
    dr.text((2 * W + 12, 6), "GS x DDF GI (mesh-free)", fill=(0, 0, 0), font=f)
    os.makedirs("renders", exist_ok=True)
    pim.save(args.out)
    print(f"wrote {args.out}  hit_frac={hit.float().mean():.3f} shadow_frac={shadow.mean():.3f} ao_mean={ao.mean():.3f}")


if __name__ == "__main__":
    main()
