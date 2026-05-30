"""S2: turntable video of the mesh-free DDF GI composite on the garden scene.

Loads the GS + DDF once, iterates azimuth in steps of 15deg (24 frames), and
writes a 3-panel sheet per frame (GS only | DDF GI map | GS x DDF GI). The
per-frame layout matches scripts/gi_garden.py exactly so the video reads like
a turntable of the static composite figure (see the paper's real-scene GI section).

Output: supplemental/gi_garden_frames/frame_NNN.png (24 files). Caller can
encode with ffmpeg if available.
"""
import os, sys, math, argparse, time, importlib.util
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, ".")
from gsplat import rasterization
from src.train_ddf import build_model
from src.ddf_gi_render import ddf_occluded

# Reuse the helpers from gi_garden.py (sibling script, not a package).
_gg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gi_garden.py")
_spec = importlib.util.spec_from_file_location("gi_garden", _gg_path)
_gg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_gg)
c2w_zup, pixel_rays, render_gs, depth_normals, hemi = (
    _gg.c2w_zup, _gg.pixel_rays, _gg.render_gs, _gg.depth_normals, _gg.hemi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/garden_obj/ddf_surf/ddf_005000.pt")
    ap.add_argument("--gs", default="runs/garden_obj/gaussians.pt")
    ap.add_argument("--image_size", type=int, default=640)
    ap.add_argument("--elev", type=float, default=32.0)
    ap.add_argument("--radius", type=float, default=2.6)
    ap.add_argument("--ambient", type=float, default=0.32)
    ap.add_argument("--ao_strength", type=float, default=0.8)
    ap.add_argument("--n_frames", type=int, default=24)
    ap.add_argument("--n_ao", type=int, default=24,
                    help="AO samples per pixel (24 matches scripts/gi_garden.py).")
    ap.add_argument("--out_dir", default="supplemental/gi_garden_frames")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    H = W = args.image_size
    fx = fy = 0.9 * W
    K = torch.tensor([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]],
                     dtype=torch.float32, device=dev)

    os.makedirs(args.out_dir, exist_ok=True)

    # Load once.
    print(f"loading GS {args.gs} and DDF {args.ckpt}")
    gs = torch.load(args.gs, map_location=dev, weights_only=False)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = build_model(ck["cfg"]).to(dev).eval()
    model.load_state_dict(ck["model"])

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19)
    except Exception:
        font = ImageFont.load_default()

    azims = [i * (360.0 / args.n_frames) for i in range(args.n_frames)]
    light_world = torch.tensor([0.45, 0.80, 0.40], device=dev)
    light_world = light_world / light_world.norm()
    pad = 30

    t_total = time.time()
    for idx, az in enumerate(azims):
        t0 = time.time()
        c2w = c2w_zup(args.elev, az, args.radius, 0.0, up_sign=-1.0)  # flip_up
        o, d = pixel_rays(c2w, K, H, W, dev)
        with torch.no_grad():
            gs_rgb, ed, alpha = render_gs(gs, c2w, K, H, W, dev)

            fwd = torch.from_numpy(c2w[:3, 2]).to(dev)
            hit = alpha.reshape(-1) > 0.5
            t = ed.reshape(-1) / (d @ fwd).clamp_min(1e-6)
            P = o + t.unsqueeze(-1) * d
            n = depth_normals(P, hit, H, W)
            flip_n = (n * d).sum(-1) > 0
            n = n.clone()
            n[flip_n] = -n[flip_n]

            to_light = light_world
            ndotl = (n * to_light).sum(-1).clamp(min=0.0)
            shadow = torch.zeros(hit.shape[0], device=dev)
            ao = torch.zeros(hit.shape[0], device=dev)
            if hit.any():
                hi = hit.nonzero(as_tuple=True)[0]
                so = P[hi] + 0.05 * n[hi]
                shadow[hi] = ddf_occluded(
                    model, so,
                    to_light.expand(hi.shape[0], 3).contiguous(),
                    torch.full((hi.shape[0],), 6.0, device=dev),
                    t_self=0.05).float()
                of = torch.zeros(hi.shape[0], device=dev)
                # AO hemi takes a deterministic seed; vary it per frame so the
                # AO noise pattern doesn't sit perfectly still on the object as
                # the camera rotates (very minor visual nicety).
                for sd in hemi(n[hi], hi.shape[0], args.n_ao, dev, seed=idx):
                    of += ddf_occluded(
                        model, so, sd,
                        torch.full((hi.shape[0],), 0.5, device=dev),
                        t_self=0.05).float()
                ao[hi] = of / args.n_ao

            alb = gs_rgb.reshape(-1, 3)
            lit = args.ambient + (1 - args.ambient) * ndotl * (1 - shadow)
            ao_fac = 1 - args.ao_strength * ao
            gi = alb * (lit * ao_fac).unsqueeze(-1)
            bg = torch.ones_like(alb)
            gi_factor = ((1 - shadow) * (1 - args.ao_strength * ao)).clamp(0, 1)
            gi_map = gi_factor.unsqueeze(-1).expand(-1, 3)
            beauty = torch.where((alpha.reshape(-1) > 0.5).unsqueeze(-1),
                                 alb, bg).reshape(H, W, 3)
            gi_map_img = torch.where(hit.unsqueeze(-1), gi_map,
                                     bg).reshape(H, W, 3)
            gi_img = torch.where(hit.unsqueeze(-1), gi, bg).reshape(H, W, 3)
            panels = torch.cat([beauty, gi_map_img, gi_img], dim=1).clamp(0, 1)
            arr = (panels.cpu().numpy() * 255).astype(np.uint8)

        sheet = np.full((pad + H, 3 * W, 3), 255, np.uint8)
        sheet[pad:] = arr
        pim = Image.fromarray(sheet)
        dr = ImageDraw.Draw(pim)
        dr.text((12, 6), "GS only (real captured scene)",
                fill=(0, 0, 0), font=font)
        dr.text((W + 12, 6), "DDF GI map (occlusion, 1 eval/ray)",
                fill=(0, 0, 0), font=font)
        dr.text((2 * W + 12, 6), "GS x DDF GI (mesh-free)",
                fill=(0, 0, 0), font=font)
        out = os.path.join(args.out_dir, f"frame_{idx:03d}.png")
        pim.save(out)
        print(f"[{idx+1}/{args.n_frames}] az={az:6.1f}deg "
              f"hit={hit.float().mean():.3f} shadow={shadow.mean():.3f} "
              f"ao={ao.mean():.3f}  ({time.time()-t0:.2f}s) -> {out}")

    print(f"done. {args.n_frames} frames in {time.time()-t_total:.1f}s "
          f"to {args.out_dir}")


if __name__ == "__main__":
    main()
