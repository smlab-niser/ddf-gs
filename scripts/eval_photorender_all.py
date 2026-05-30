"""End-to-end eval: Chamfer + photometric (PSNR) for a photo-render ckpt.

Wraps stage3_chamfer + eval_photorender into a single driver and dumps a
SUMMARY.md comparing the new DDF to KS-only and NeuS @ 30k for one object.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gs_path", default=None)
    ap.add_argument("--views_dir", default=None)
    ap.add_argument("--cuda_device", default="3")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ks_ckpt", default=None,
                    help="kitchen-sink DDF ckpt for the same obj (for PSNR comparison)")
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--n_views", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    views_dir = args.views_dir or f"runs/{args.obj}/views"
    gs_path = args.gs_path or f"runs/{args.obj}/gaussians.pt"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    env["PYTHONPATH"] = "."

    # 1. Chamfer.
    print("==> running Chamfer eval")
    cmd_cd = [
        sys.executable, "scripts/stage3_chamfer.py",
        "--obj", args.obj, "--ddf_ckpt", args.ckpt,
        "--gs_path", gs_path,
        "--out_dir", str(out_dir / "stage3"),
    ]
    print(" ".join(cmd_cd))
    subprocess.check_call(cmd_cd, env=env)

    # 2. PSNR eval on the new ckpt.
    print("==> running photometric eval")
    cmd_psnr = [
        sys.executable, "scripts/eval_photorender.py",
        "--ckpt", args.ckpt, "--views_dir", views_dir,
        "--out_dir", str(out_dir / "psnr"),
        "--n_samples", str(args.n_samples),
        "--n_views", str(args.n_views),
    ]
    print(" ".join(cmd_psnr))
    subprocess.check_call(cmd_psnr, env=env)

    # 3. Aggregate.
    chamfer = json.loads((out_dir / "stage3" / "metrics.json").read_text())
    psnr = json.loads((out_dir / "psnr" / "psnr.json").read_text())

    summary = {
        "obj": args.obj,
        "photorender": {
            "chamfer_mean": chamfer["chamfer_mean"],
            "chamfer_median": chamfer["chamfer_median"],
            "psnr_mean": psnr["mean"],
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
