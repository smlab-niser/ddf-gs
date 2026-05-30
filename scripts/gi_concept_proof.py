"""GI concept proof: hard directional shadow on bull, DDF vs embree reference.

Milestone gate for the TVCG GI track. Renders the same shadowed view with:
  - DDF oracle (sphere-trace + 1-eval shadow visibility)
  - embree mesh oracle (ground truth)
identical shading math (callback injection). Reports PSNR/SSIM/shadow-IoU + time.
"""
import os, sys, math, time, json, argparse
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from src.train_ddf import build_model
from src.ddf_gi_render import (
    ddf_sphere_trace, ddf_occluded, ddf_normals, MeshOracle,
    render_shadows, render_ao,
)
from src.gtmesh_supervisor import GTMeshSupervisor
from sphere_trace_extract import pixel_rays, look_at_c2w, intersect_sphere


def psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    return 50.0 if mse < 1e-10 else -10 * math.log10(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="bull")
    ap.add_argument("--ckpt", default="runs/bull_ddf_gtmesh/ddf_final.pt")
    ap.add_argument("--gs_path", default="runs/bull_50k/gaussians.pt")
    ap.add_argument("--gt_mesh", default="data/gso/Schleich_Hereford_Bull/meshes/model.obj")
    ap.add_argument("--gso_rotate", type=lambda s: s.lower() in ("1", "true", "yes"),
                    default=False, help="match the DDF's training supervisor.gso_rotate (gtmesh bull=false)")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=30.0)
    ap.add_argument("--radius", type=float, default=2.5)
    ap.add_argument("--fov", type=float, default=50.0)
    ap.add_argument("--mode", default="shadow", choices=["shadow", "ao"])
    ap.add_argument("--n_ao", type=int, default=32)
    ap.add_argument("--out_dir", default="renders/gi")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---- load DDF ----
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = build_model(ck["cfg"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    print(f"DDF loaded: {type(model).__name__}, {sum(p.numel() for p in model.parameters()):,} params")

    # ---- load embree mesh in the SAME frame ----
    gs = torch.load(args.gs_path, map_location="cpu", weights_only=False)
    mc = gs["mesh_center"].numpy().astype(np.float32)
    ms = float(gs["mesh_scale"].item())
    sup = GTMeshSupervisor(args.gt_mesh, mc, ms, device=dev,
                           march_ratio=0.0, gso_rotate=args.gso_rotate)
    mesh_oracle = MeshOracle(sup.mesh, device=dev)

    # ---- camera (single view) ----
    eye = np.array([
        args.radius * math.cos(math.radians(args.elev)) * math.sin(math.radians(args.azim)),
        args.radius * math.sin(math.radians(args.elev)),
        args.radius * math.cos(math.radians(args.elev)) * math.cos(math.radians(args.azim)),
    ], dtype=np.float32)
    c2w = torch.from_numpy(look_at_c2w(eye)).to(dev)
    origins, dirs = pixel_rays(c2w, args.image_size, args.fov, dev)
    origins = origins.reshape(-1, 3); dirs = dirs.reshape(-1, 3)

    # ---- light ----
    light_dir = torch.tensor([0.4, -0.8, -0.3], device=dev)
    light_dir = light_dir / light_dir.norm()

    # ---- DDF oracle callbacks ----
    # Jump-start camera rays to the bbox sphere (hash grid extrapolates garbage
    # outside [-bbox_half, bbox_half]; sphere_trace_extract does the same).
    bbox_half = ck["cfg"]["model"].get("bbox_half", 1.2)
    def ddf_intersect(o, d):
        t_enter = intersect_sphere(o, d, bbox_half * 1.1)
        o_js = o + t_enter.unsqueeze(-1) * d
        hit, x, t_local, vis = ddf_sphere_trace(model, o_js, d, bbox_half=bbox_half)
        t = t_enter + t_local  # distance from the true camera origin
        return hit, x, t, None
    def ddf_occ(o, d, md):
        return ddf_occluded(model, o, d, md)
    def ddf_nrm(x, d):
        return ddf_normals(model, x, d)

    # ---- embree oracle callbacks ----
    def mesh_intersect(o, d):
        return mesh_oracle.intersect(o, d)
    def mesh_occ(o, d, md):
        return mesh_oracle.occluded(o, d, md)
    # mesh normals come from intersect (geom face normals); normal_fn=None uses them

    isz = args.image_size

    def run(intersect_fn, occ_fn, nrm_fn, tag):
        torch.cuda.synchronize(); t0 = time.time()
        if args.mode == "shadow":
            r = render_shadows(intersect_fn, occ_fn, nrm_fn, origins, dirs, light_dir, device=dev)
        else:
            r = render_ao(intersect_fn, occ_fn, nrm_fn, origins, dirs,
                          n_ao=args.n_ao, device=dev)
        torch.cuda.synchronize(); dt = time.time() - t0
        img = r["rgb"].reshape(isz, isz, 3).clamp(0, 1)
        Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)).save(out / f"{args.obj}_{args.mode}_{tag}.png")
        return r, img, dt

    # DDF uses its own normals; mesh uses geometric face normals (normal_fn=None)
    r_ddf, img_ddf, t_ddf = run(ddf_intersect, ddf_occ, ddf_nrm, "ddf")
    r_mesh, img_mesh, t_mesh = run(mesh_intersect, mesh_occ, None, "mesh")

    # ---- metrics (foreground = mesh hit) ----
    fg = r_mesh["hit"].reshape(isz, isz)
    p = psnr(img_ddf, img_mesh)
    # shadow IoU on foreground
    sd_ddf = r_ddf["shadow"].reshape(isz, isz) if args.mode == "shadow" else None
    iou_str = ""
    if args.mode == "shadow":
        sd_mesh = r_mesh["shadow"].reshape(isz, isz)
        inter = (sd_ddf & sd_mesh & fg).sum().item()
        union = ((sd_ddf | sd_mesh) & fg).sum().item()
        iou = inter / max(union, 1)
        iou_str = f"  shadow-IoU={iou:.3f}"

    # primary-depth sanity (DDF vs mesh agreement on foreground)
    dd = r_ddf["depth"].reshape(isz, isz)[fg]
    dm = r_mesh["depth"].reshape(isz, isz)[fg]
    depth_l1 = (dd - dm).abs().median().item() if fg.any() else float("nan")

    # error image
    err = (img_ddf - img_mesh).abs().mean(-1)
    Image.fromarray((err.cpu().numpy() * 255 * 3).clip(0, 255).astype(np.uint8)).save(
        out / f"{args.obj}_{args.mode}_error.png")
    # side-by-side
    sbs = torch.cat([img_ddf, img_mesh, err.unsqueeze(-1).expand(-1, -1, 3).clamp(0, 1)], dim=1)
    Image.fromarray((sbs.cpu().numpy() * 255).astype(np.uint8)).save(
        out / f"{args.obj}_{args.mode}_sidebyside.png")

    print(f"\n=== {args.obj} {args.mode} : DDF vs embree ===")
    print(f"  PSNR={p:.2f} dB{iou_str}")
    print(f"  primary-depth median L1 (DDF vs mesh, fg) = {depth_l1:.4f}  (sanity: should be < ~0.1)")
    print(f"  render time: DDF={t_ddf*1000:.1f} ms  mesh={t_mesh*1000:.1f} ms  ratio={t_mesh/max(t_ddf,1e-6):.2f}x")
    print(f"  saved {out}/{args.obj}_{args.mode}_{{ddf,mesh,error,sidebyside}}.png")
    json.dump({"psnr": p, "depth_l1": depth_l1, "t_ddf_ms": t_ddf*1000,
               "t_mesh_ms": t_mesh*1000, "mode": args.mode},
              open(out / f"{args.obj}_{args.mode}_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
