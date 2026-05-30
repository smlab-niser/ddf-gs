"""E3 hero figure: GS appearance relit with DDF-traced GI (self-shadow + AO),
with an embree ground-truth panel. Secondary rays are cast from SURFACE points
(offset along the normal) = the in-distribution regime the oracle was trained on
(the E4 30 dB result), so the DDF panel closely matches embree.

(Floor cast-shadows were tried and abandoned: floor-point ray origins are
out-of-distribution for the oracle — it false-occludes ~65% of the floor vs
embree's ~9%, untunable. Surface-anchored shadow/AO is the faithful regime.)

Geometry + normals come from embree (clean, noise-free) and are SHARED by all
three panels; the per-pixel ALBEDO is the gsplat RGB (the GS appearance); only
the shadow/AO occlusion oracle differs (DDF vs embree), so any visible difference
is purely the DDF-vs-embree secondary-ray gap.

Per object, three labelled panels:
  1. GS only (no GI)       — Lambertian on GS albedo, no shadow/AO.
  2. GS + DDF GI (ours)     — + DDF self-shadow + DDF ambient occlusion.
  3. GS + embree GI (GT)    — + embree self-shadow + embree AO (ground truth).

Run:  PYTHONPATH=. CUDA_VISIBLE_DEVICES=<g> python scripts/gi_hero_figure.py
"""
import os, sys, math, argparse
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, ".")
from src.train_ddf import build_model
from src.ddf_gi_render import ddf_occluded, MeshOracle, orient_normals
from src.gtmesh_supervisor import GTMeshSupervisor
from gsplat import rasterization

OBJS = [
    ("car",    "Vtech_Cruise_Learn_Car_25_Years"),
    ("horse",  "Breyer_Horse_Of_The_Year_2015"),
    ("spino",  "Schleich_Spinosaurus_Action_Figure"),
    ("teapot", "Threshold_Porcelain_Teapot_White"),
    ("lion",   "Schleich_Lion_Action_Figure"),
    ("bull",   "Schleich_Hereford_Bull"),
]
CELL = 420  # square cell; each object is cropped to its bbox and fit into a cell


def fit_cell(a):
    """Resize a cropped object panel to fill a CELL x CELL white cell (centered)."""
    h, w = a.shape[:2]
    s = CELL / max(h, w)
    im = np.asarray(Image.fromarray(a).resize((max(1, int(w * s)), max(1, int(h * s)))))
    cell = np.full((CELL, CELL, 3), 255, np.uint8)
    oy, ox = (CELL - im.shape[0]) // 2, (CELL - im.shape[1]) // 2
    cell[oy:oy + im.shape[0], ox:ox + im.shape[1]] = im
    return cell


def build_c2w(elev_deg, azim_deg, radius):
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    eye = np.array([radius*math.cos(e)*math.sin(a), radius*math.sin(e),
                    radius*math.cos(e)*math.cos(a)], dtype=np.float32)
    fwd = -eye/(np.linalg.norm(eye)+1e-8)
    up = np.array([0., 1., 0.], dtype=np.float32)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)+1e-8
    down = -np.cross(fwd, right)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right; c2w[:3, 1] = down; c2w[:3, 2] = fwd; c2w[:3, 3] = eye
    return c2w


def pixel_rays(c2w, K, H, W, device):
    ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                            torch.arange(W, device=device, dtype=torch.float32),
                            indexing="ij")
    x = (xs + 0.5 - K[0, 2]) / K[0, 0]
    y = (ys + 0.5 - K[1, 2]) / K[1, 1]
    dcam = torch.stack([x, y, torch.ones_like(x)], -1).reshape(-1, 3)
    R = torch.from_numpy(c2w[:3, :3]).to(device)
    dirs = (dcam @ R.T); dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    o = torch.from_numpy(c2w[:3, 3]).to(device).expand_as(dirs)
    return o.contiguous(), dirs.contiguous()


@torch.no_grad()
def render_gs(gs, c2w, K, H, W, device):
    means = gs["means"].to(device); quats = gs["quats"].to(device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = gs["scales"].to(device).exp(); opac = gs["opacities"].to(device).sigmoid()
    colors = gs["colors"].to(device).sigmoid()
    w2c = torch.linalg.inv(torch.from_numpy(c2w).to(device)).unsqueeze(0)
    renders, alphas, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
        viewmats=w2c, Ks=K.unsqueeze(0), width=W, height=H, sh_degree=None,
        render_mode="RGB")
    return renders[0, ..., :3].clamp(0, 1), alphas[0, ..., 0]


def hemisphere_dirs(n_h, M, n_ao, device, seed=0):
    up = torch.where(n_h[:, 1:2].abs() < 0.9,
                     torch.tensor([0., 1., 0.], device=device).expand(M, 3),
                     torch.tensor([1., 0., 0.], device=device).expand(M, 3))
    tang = torch.cross(up, n_h, dim=-1); tang = tang / tang.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    bit = torch.cross(n_h, tang, dim=-1)
    g = torch.Generator(device=device).manual_seed(seed)
    for _ in range(n_ao):
        u1 = torch.rand(M, device=device, generator=g); u2 = torch.rand(M, device=device, generator=g)
        r = u1.sqrt(); th = 2 * math.pi * u2
        dd = (r*torch.cos(th)).unsqueeze(-1)*tang + (r*torch.sin(th)).unsqueeze(-1)*bit + \
             (1-u1).clamp(min=0).sqrt().unsqueeze(-1)*n_h
        yield dd / dd.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def shade(gs_rgb, intersect_fn, occ_fn, o, d, light_dir, H, W, device, *,
          ambient=0.30, off=0.05, ao_radius=0.6, n_ao=32, ao_strength=0.8,
          albedo_gain=1.7):
    """Relight GS albedo with embree-clean geometry/normals + (optional) GI."""
    hit, P, t, nrm = intersect_fn(o, d)
    nrm = orient_normals(nrm, d)
    to_light = (-light_dir); to_light = to_light / to_light.norm()
    ndotl = (nrm * to_light).sum(-1).clamp(min=0.0)
    shadow = torch.zeros(hit.shape[0], device=device)
    ao_occ = torch.zeros(hit.shape[0], device=device)
    if occ_fn is not None and hit.any():
        hi = hit.nonzero(as_tuple=True)[0]
        so = P[hi] + off * nrm[hi]
        shadow[hi] = occ_fn(so, to_light.expand(hi.shape[0], 3).contiguous(),
                            torch.full((hi.shape[0],), 10.0, device=device)).float()
        of = torch.zeros(hi.shape[0], device=device)
        for sd in hemisphere_dirs(nrm[hi], hi.shape[0], n_ao, device):
            of += occ_fn(so, sd, torch.full((hi.shape[0],), ao_radius, device=device)).float()
        ao_occ[hi] = of / n_ao
    lit = ambient + (1 - ambient) * ndotl * (1 - shadow)
    ao_fac = 1 - ao_strength * ao_occ
    alb = (gs_rgb.reshape(-1, 3) * albedo_gain).clamp(0, 1)
    col = alb * (lit * ao_fac).unsqueeze(-1)
    out = torch.where(hit.unsqueeze(-1), col, torch.ones_like(alb))
    return out.reshape(H, W, 3)


def label_sheet(rows, H, W):
    """Stack rows with a large, centered column-label header. No row labels."""
    pad_top = 58
    n = len(rows)
    sheet = np.full((pad_top + n * H, 3 * W, 3), 255, np.uint8)
    for r, row in enumerate(rows):
        sheet[pad_top + r * H: pad_top + (r + 1) * H, :] = row
    img = Image.fromarray(sheet)
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 38)
    except Exception:
        font = ImageFont.load_default()
    for c, lab in enumerate(COL_LABELS):
        bb = dr.textbbox((0, 0), lab, font=font)
        dr.text((c * W + (W - (bb[2] - bb[0])) // 2, 12), lab, fill=(0, 0, 0), font=font)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--elev", type=float, default=28.0)
    ap.add_argument("--azim", type=float, default=40.0)
    ap.add_argument("--radius", type=float, default=2.4)
    ap.add_argument("--out", default="renders/gi_hero.png")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    H = W = args.image_size
    fx = fy = 0.85 * W
    K = torch.tensor([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]], dtype=torch.float32, device=dev)
    c2w = build_c2w(args.elev, args.azim, args.radius)
    o, d = pixel_rays(c2w, K, H, W, dev)
    light_dir = torch.tensor([0.62, -0.32, 0.42], device=dev); light_dir /= light_dir.norm()

    rows, names = [], []
    for oid, mesh_name in OBJS:
        gs_path, ckpt = f"runs/{oid}/gaussians.pt", f"runs/{oid}_ddf_gtmesh/ddf_final.pt"
        mesh_path = f"data/gso/{mesh_name}/meshes/model.obj"
        if not all(os.path.exists(p) for p in (gs_path, ckpt, mesh_path)):
            print(f"[{oid}] missing asset, skip"); continue
        gs = torch.load(gs_path, map_location="cpu", weights_only=False)
        mc = gs["mesh_center"].numpy().astype(np.float32); ms = float(gs["mesh_scale"].item())
        gs_rgb, alpha = render_gs(gs, c2w, K, H, W, dev)

        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        model = build_model(ck["cfg"]).to(dev).eval(); model.load_state_dict(ck["model"])
        sup = GTMeshSupervisor(mesh_path, mc, ms, device=dev, march_ratio=0.0, gso_rotate=False)
        mo = MeshOracle(sup.mesh, device=dev)
        ddf_occ = lambda so, sd, md: ddf_occluded(model, so, sd, md, t_self=0.04)
        mesh_occ = lambda so, sd, md: mo.occluded(so, sd, md, t_self=0.04)

        p_gs = shade(gs_rgb, mo.intersect, None, o, d, light_dir, H, W, dev)
        p_ddf = shade(gs_rgb, mo.intersect, ddf_occ, o, d, light_dir, H, W, dev)
        p_ref = shade(gs_rgb, mo.intersect, mesh_occ, o, d, light_dir, H, W, dev)
        # crop the three panels to the object bbox so it fills the cell (no whitespace)
        am = (alpha.detach().cpu().numpy() > 0.5)
        ys, xs = np.where(am); mgn = 10
        if len(ys):
            y0, y1 = max(0, ys.min() - mgn), min(H, ys.max() + 1 + mgn)
            x0, x1 = max(0, xs.min() - mgn), min(W, xs.max() + 1 + mgn)
        else:
            y0, y1, x0, x1 = 0, H, 0, W
        cells = [fit_cell((p.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[y0:y1, x0:x1])
                 for p in (p_gs, p_ddf, p_ref)]
        rows.append(np.concatenate(cells, axis=1))
        print(f"[{oid}] rendered", flush=True)
        del model; torch.cuda.empty_cache()

    if not rows:
        print("no rows"); return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out)
    print(f"wrote {args.out} ({len(rows)} obj x 3 panels, cropped to fill, no baked labels)")


if __name__ == "__main__":
    main()
