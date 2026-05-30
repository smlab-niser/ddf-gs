"""E1 decisive table: shadow-ray agreement of DDF oracles vs GT-mesh embree.

For each object, sample N surface points on the GT mesh, cast shadow rays toward
a set of light directions, and measure the fraction of rays where the DDF's
1-eval occlusion test agrees with embree's GT-mesh occlusion.

Three oracles compared (the paper's logical lynchpin, plan E1):
  (a) gtmesh-omnidir  [upper bound, trained from the mesh]
  (b) NeuS-omnidir    [the mesh-free contribution: images -> NeuS -> DDF]
  (c) neus_v3         [frustum-only baseline that fails on secondary rays]

Success criterion: (b) >> (c) and (b) approaches (a).

Run:  PYTHONPATH=. CUDA_VISIBLE_DEVICES=<g> python scripts/e1_shadow_agreement.py
"""
import os, sys, json, argparse
import numpy as np, torch, trimesh
sys.path.insert(0, ".")
from src.train_ddf import build_model
from src.ddf_gi_render import ddf_occluded, MeshOracle
from src.gtmesh_supervisor import GTMeshSupervisor

OBJS = {
    "bull":   "data/gso/Schleich_Hereford_Bull/meshes/model.obj",
    "mug":    "data/gso/ACE_Coffee_Mug_Kristen_16_oz_cup/meshes/model.obj",
    "turtle": "data/gso/Vtech_Roll_Learn_Turtle/meshes/model.obj",
    "lion":   "data/gso/Schleich_Lion_Action_Figure/meshes/model.obj",
}
LIGHTS = torch.tensor([
    [0.4, -0.8, -0.3], [-0.5, -0.7, 0.4], [0.0, -1.0, 0.0],
    [0.6, -0.3, 0.7], [-0.3, -0.6, -0.7],
], dtype=torch.float32)


def agreement(model, sho, tl, mocc, dev, far):
    docc = ddf_occluded(model, sho, tl, torch.full((sho.shape[0],), far, device=dev))
    return (docc == mocc).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--far", type=float, default=10.0)
    ap.add_argument("--out", default="runs/e1_shadow_agreement.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)

    results = {}
    for obj, mesh_path in OBJS.items():
        if not os.path.exists(mesh_path):
            print(f"[{obj}] mesh missing, skip"); continue
        gs = torch.load(f"runs/{obj}/gaussians.pt", map_location="cpu", weights_only=False)
        mc = gs["mesh_center"].numpy().astype(np.float32); ms = float(gs["mesh_scale"].item())
        sup = GTMeshSupervisor(mesh_path, mc, ms, device=dev, march_ratio=0.0, gso_rotate=False)
        gt_oracle = MeshOracle(sup.mesh, device=dev)

        pts_np, _ = trimesh.sample.sample_surface(sup.mesh, args.n)
        pts = torch.from_numpy(np.asarray(pts_np, np.float32)).to(dev)

        # build the full shadow-ray set across all light dirs
        sho_all, tl_all = [], []
        for L in LIGHTS:
            L = (L / L.norm()).to(dev)
            tl = (-L).expand(pts.shape[0], 3)
            sho_all.append(pts + 0.05 * tl); tl_all.append(tl)
        sho = torch.cat(sho_all); tl = torch.cat(tl_all)
        mocc = gt_oracle.occluded(sho, tl, torch.full((sho.shape[0],), args.far, device=dev))
        shadow_rate = 100 * mocc.float().mean().item()

        row = {"shadow_rate_pct": shadow_rate, "n_rays": sho.shape[0]}
        ck_map = {
            "gtmesh_omnidir": f"runs/{obj}_ddf_gtmesh/ddf_final.pt",
            "neus_omnidir":   f"runs/{obj}_ddf_neus_omnidir/ddf_final.pt",
            "neus_v3":        f"runs/{obj}_ddf_neus_v3/ddf_final.pt",
        }
        for name, cp in ck_map.items():
            if not os.path.exists(cp):
                row[name] = None; continue
            ck = torch.load(cp, map_location=dev, weights_only=False)
            m = build_model(ck["cfg"]).to(dev).eval(); m.load_state_dict(ck["model"])
            row[name] = round(100 * agreement(m, sho, tl, mocc, dev, args.far), 1)
            del m; torch.cuda.empty_cache()
        results[obj] = row
        print(f"[{obj}] shadow_rate={shadow_rate:4.1f}%  "
              f"gtmesh={row['gtmesh_omnidir']}  neus_omni={row['neus_omnidir']}  "
              f"neus_v3={row['neus_v3']}", flush=True)

    # aggregate (objects with all three present)
    full = [r for r in results.values()
            if r.get("gtmesh_omnidir") and r.get("neus_omnidir")]
    if full:
        agg = {k: round(float(np.mean([r[k] for r in full if r.get(k) is not None])), 1)
               for k in ("gtmesh_omnidir", "neus_omnidir", "neus_v3")}
        results["_mean"] = agg
        print(f"[MEAN n={len(full)}] gtmesh={agg['gtmesh_omnidir']}  "
              f"neus_omni={agg['neus_omnidir']}  neus_v3={agg['neus_v3']}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
