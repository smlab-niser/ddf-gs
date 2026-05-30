"""Relighting strip: one real-scene GS object relit under several light directions
using the DDF as the light-visibility oracle (self-shadow + AO, 1 eval/ray). The
shading and shadows MOVE with the light — the secondary-ray capability GS lacks.

Panel 0 = GS only (flat appearance, no light response). Panels 1..N = the same GS
appearance relit per light direction via DDF occlusion. Mesh-free.

Run: PYTHONPATH=. CUDA_VISIBLE_DEVICES=<g> python scripts/relight_garden.py \
       --ckpt runs/<scene>_obj/ddf_surf/ddf_005000.pt --gs runs/<scene>_obj/gaussians.pt --flip_up
"""
import os, sys, math, argparse
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, ".")
from gsplat import rasterization
from src.train_ddf import build_model
from src.ddf_gi_render import ddf_occluded


def c2w_orbit(elev, azim, r, up_sign=1.0):
    e, a = math.radians(elev), math.radians(azim)
    eye = np.array([r*math.cos(e)*math.sin(a), r*math.sin(e), r*math.cos(e)*math.cos(a)], np.float32)
    fwd = -eye/np.linalg.norm(eye); up = np.array([0, up_sign, 0], np.float32)
    right = np.cross(fwd, up); right /= np.linalg.norm(right); down = -np.cross(fwd, right)
    c2w = np.eye(4, dtype=np.float32); c2w[:3, 0] = right; c2w[:3, 1] = down; c2w[:3, 2] = fwd; c2w[:3, 3] = eye
    return c2w


def pixel_rays(c2w, K, H, W, dev):
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    x = (xs+0.5-K[0, 2])/K[0, 0]; y = (ys+0.5-K[1, 2])/K[1, 1]
    dcam = torch.stack([x, y, torch.ones_like(x)], -1).reshape(-1, 3)
    R = torch.from_numpy(c2w[:3, :3]).to(dev); d = (dcam @ R.T); d = d/d.norm(dim=-1, keepdim=True)
    return torch.from_numpy(c2w[:3, 3]).to(dev).expand_as(d).contiguous(), d.contiguous()


@torch.no_grad()
def render_gs(gs, c2w, K, H, W, dev):
    means = gs["means"].to(dev); q = gs["quats"].to(dev); q = q/q.norm(dim=-1, keepdim=True)
    sc = gs["scales"].to(dev).exp(); op = gs["opacities"].to(dev).sigmoid(); col = gs["colors"].to(dev).sigmoid()
    w2c = torch.linalg.inv(torch.from_numpy(c2w).to(dev)).unsqueeze(0)
    ren, al, _ = rasterization(means=means, quats=q, scales=sc, opacities=op, colors=col,
                               viewmats=w2c, Ks=K.unsqueeze(0), width=W, height=H, sh_degree=None, render_mode="RGB+ED")
    return ren[0, ..., :3].clamp(0, 1), ren[0, ..., 3], al[0, ..., 0]


def normals(P, H, W):
    Pg = P.reshape(H, W, 3); dx = torch.zeros_like(Pg); dy = torch.zeros_like(Pg)
    dx[:, 1:-1] = Pg[:, 2:]-Pg[:, :-2]; dy[1:-1, :] = Pg[2:, :]-Pg[:-2, :]
    n = torch.cross(dx, dy, dim=-1); return (n/n.norm(dim=-1, keepdim=True).clamp_min(1e-8)).reshape(-1, 3)


def hemi(n_h, M, n_ao, dev, seed=0):
    up = torch.where(n_h[:, 2:3].abs() < 0.9, torch.tensor([0., 0., 1.], device=dev).expand(M, 3),
                     torch.tensor([1., 0., 0.], device=dev).expand(M, 3))
    t = torch.cross(up, n_h, dim=-1); t = t/t.norm(dim=-1, keepdim=True).clamp_min(1e-8); b = torch.cross(n_h, t, dim=-1)
    g = torch.Generator(device=dev).manual_seed(seed)
    for _ in range(n_ao):
        u1 = torch.rand(M, device=dev, generator=g); u2 = torch.rand(M, device=dev, generator=g)
        r = u1.sqrt(); th = 2*math.pi*u2
        d = (r*torch.cos(th)).unsqueeze(-1)*t + (r*torch.sin(th)).unsqueeze(-1)*b + (1-u1).clamp(min=0).sqrt().unsqueeze(-1)*n_h
        yield d/d.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--gs", required=True)
    ap.add_argument("--image_size", type=int, default=560)
    ap.add_argument("--elev", type=float, default=26); ap.add_argument("--azim", type=float, default=55)
    ap.add_argument("--radius", type=float, default=2.6); ap.add_argument("--flip_up", action="store_true")
    ap.add_argument("--ambient", type=float, default=0.25); ap.add_argument("--ao_strength", type=float, default=0.6)
    ap.add_argument("--albedo_gain", type=float, default=1.0)
    ap.add_argument("--light_elev", type=float, default=35.0)
    ap.add_argument("--light_azims", default="20,70,120,170")
    ap.add_argument("--out", default="renders/relight_garden.png")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    H = W = args.image_size; fx = fy = 0.9*W
    K = torch.tensor([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]], dtype=torch.float32, device=dev)
    usign = -1.0 if args.flip_up else 1.0
    c2w = c2w_orbit(args.elev, args.azim, args.radius, usign)
    o, d = pixel_rays(c2w, K, H, W, dev)

    gs = torch.load(args.gs, map_location=dev, weights_only=False)
    gs_rgb, ed, alpha = render_gs(gs, c2w, K, H, W, dev)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = build_model(ck["cfg"]).to(dev).eval(); model.load_state_dict(ck["model"])

    fwd = torch.from_numpy(c2w[:3, 2]).to(dev)
    hit = alpha.reshape(-1) > 0.5
    t = ed.reshape(-1)/(d @ fwd).clamp_min(1e-6); P = o + t.unsqueeze(-1)*d
    n = normals(P, H, W); flip = (n*d).sum(-1) > 0; n = n.clone(); n[flip] = -n[flip]
    alb = (gs_rgb.reshape(-1, 3) * args.albedo_gain).clamp(0, 1); bg = torch.ones_like(alb)

    # AO once (fixed across lights)
    ao = torch.zeros(hit.shape[0], device=dev)
    hi = hit.nonzero(as_tuple=True)[0]; so = P[hi] + 0.05*n[hi]
    of = torch.zeros(hi.shape[0], device=dev)
    for sd in hemi(n[hi], hi.shape[0], 24, dev):
        of += ddf_occluded(model, so, sd, torch.full((hi.shape[0],), 0.5, device=dev), t_self=0.05).float()
    ao[hi] = of/24; ao_fac = 1 - args.ao_strength*ao

    def relit(to_light):
        to_light = to_light/to_light.norm()
        ndotl = (n*to_light).sum(-1).clamp(min=0.0)
        shadow = torch.zeros(hit.shape[0], device=dev)
        shadow[hi] = ddf_occluded(model, so, to_light.expand(hi.shape[0], 3).contiguous(),
                                  torch.full((hi.shape[0],), 6.0, device=dev), t_self=0.05).float()
        lit = args.ambient + (1-args.ambient)*ndotl*(1-shadow)
        col = alb*(lit*ao_fac).unsqueeze(-1)
        return torch.where(hit.unsqueeze(-1), col, bg).reshape(H, W, 3)

    # panel 0: GS only (flat appearance, headlight)
    flat = torch.where(hit.unsqueeze(-1), alb, bg).reshape(H, W, 3)
    panels = [("GS only", flat)]
    le = math.radians(args.light_elev)
    # light "above" the object = -y in the flipped frame; sweep azimuth
    ysign = -1.0 if args.flip_up else 1.0
    for az in [float(x) for x in args.light_azims.split(",")]:
        a = math.radians(az)
        L = torch.tensor([math.cos(le)*math.sin(a), ysign*math.sin(le), math.cos(le)*math.cos(a)],
                         device=dev, dtype=torch.float32)
        panels.append((f"light {int(az)}°", relit(L)))

    imgs = [(p.clamp(0, 1).cpu().numpy()*255).astype(np.uint8) for _, p in panels]
    strip = np.concatenate(imgs, axis=1)
    pad = 30; sheet = np.full((pad+H, strip.shape[1], 3), 255, np.uint8); sheet[pad:] = strip
    im = Image.fromarray(sheet); dr = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        f = ImageFont.load_default()
    for i, (lab, _) in enumerate(panels):
        dr.text((i*W+10, 6), lab, fill=(0, 0, 0), font=f)
    os.makedirs("renders", exist_ok=True); im.save(args.out)
    print(f"wrote {args.out}  ({len(panels)} panels: GS + {len(panels)-1} light dirs)  ao_mean={ao.mean():.3f}")


if __name__ == "__main__":
    main()
