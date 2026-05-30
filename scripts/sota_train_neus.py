"""Train neus-facto on an SDFStudio-formatted dataset, programmatically (no CLI),
then export a marching-cubes SDF mesh to <out_dir>/sdf_mesh.ply.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/sota_train_neus.py \
        --obj bull --max_iters 5000 --resolution 512

Logs go to <out_dir>/train.log. Wall-time is recorded to <out_dir>/wall.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Pin to GPUs 0/1 only -- sibling agent owns 2/3 for the real-world demo.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Suppress some nerfstudio FutureWarnings noise.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def build_config(data_dir: Path, output_dir: Path, exp_name: str,
                 max_iters: int, rays_per_batch: int):
    """Tuned config for our rendered datasets:
       - white background (our renders are white-on-foreground).
       - no background MLP (the scene fits in [-1, 1]^3 so no contraction).
       - overwrite near/far to match our camera radius 2.5.
    """
    from nerfstudio.configs.method_configs import method_configs
    from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
    import copy

    cfg = copy.deepcopy(method_configs["neus-facto"])
    cfg.experiment_name = exp_name
    cfg.method_name = "neus-facto"
    cfg.output_dir = output_dir
    cfg.max_num_iterations = int(max_iters) + 1
    cfg.vis = "tensorboard"
    # Aggressive eval-disable; we only care about train + final export.
    cfg.steps_per_eval_image = 10**9
    cfg.steps_per_eval_batch = 10**9
    cfg.steps_per_eval_all_images = 10**9
    cfg.steps_per_save = max(int(max_iters), 1)
    cfg.save_only_latest_checkpoint = True

    new_dp = SDFStudioDataParserConfig(
        data=data_dir,
        include_mono_prior=False,
        include_foreground_mask=False,
        scene_scale=2.0,
        auto_orient=False,
    )
    cfg.pipeline.datamanager.dataparser = new_dp
    cfg.pipeline.datamanager.train_num_rays_per_batch = int(rays_per_batch)
    cfg.pipeline.datamanager.eval_num_rays_per_batch = int(rays_per_batch)

    # Model-side fixes for our render setup ----------------------------------
    model_cfg = cfg.pipeline.model
    model_cfg.background_color = "white"
    model_cfg.background_model = "none"
    model_cfg.overwrite_near_far_plane = True
    model_cfg.near_plane = 0.5
    model_cfg.far_plane = 4.5
    # CRITICAL: default `inside_outside=True` is for indoor scenes -- it flips
    # the SDF sign so the inside of the room is positive. For object-centric
    # reconstruction the sphere init must point *outwards* (SDF > 0 outside
    # the object). Without this, the mesh degenerates to a tiny sphere at
    # the origin where the inverted SDF crosses zero.
    model_cfg.sdf_field.inside_outside = False

    # Disable viewer (headless).
    if hasattr(cfg, "viewer"):
        cfg.viewer.quit_on_train_completion = True

    return cfg


def export_marching_cubes(load_config: Path, out_mesh: Path,
                          resolution: int = 512,
                          aabb_min=(-1.0, -1.0, -1.0),
                          aabb_max=(1.0, 1.0, 1.0)):
    """Run marching-cubes export equivalent to `ns-export marching-cubes`
    but without texturing (we just need the bare mesh for Chamfer).
    """
    from typing import cast
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.exporter.marching_cubes import generate_mesh_with_multires_marching_cubes
    from nerfstudio.fields.sdf_field import SDFField

    _, pipeline, _, _ = eval_setup(load_config)
    assert hasattr(pipeline.model.config, "sdf_field"), "Model has no SDF field"

    multi_res_mesh = generate_mesh_with_multires_marching_cubes(
        geometry_callable_field=lambda x: cast(SDFField, pipeline.model.field)
            .forward_geonetwork(x)[:, 0]
            .contiguous(),
        resolution=resolution,
        bounding_box_min=tuple(float(v) for v in aabb_min),
        bounding_box_max=tuple(float(v) for v in aabb_max),
        isosurface_threshold=0.0,
        coarse_mask=None,
    )
    out_mesh.parent.mkdir(parents=True, exist_ok=True)
    multi_res_mesh.export(out_mesh)
    return out_mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--data_dir", default=None,
                    help="default runs/sota_comparison/<obj>/sdf_data")
    ap.add_argument("--out_dir", default=None,
                    help="default runs/sota_comparison/<obj>/neus")
    ap.add_argument("--max_iters", type=int, default=5000)
    ap.add_argument("--rays_per_batch", type=int, default=1024)
    ap.add_argument("--resolution", type=int, default=512,
                    help="marching cubes resolution; must be divisible by 512.")
    ap.add_argument("--aabb_half", type=float, default=1.0)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_export", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(f"runs/sota_comparison/{args.obj}/sdf_data")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"runs/sota_comparison/{args.obj}/neus")
    if not (data_dir / "meta_data.json").exists():
        raise SystemExit(f"missing meta_data.json in {data_dir}; run sota_make_sdfstudio_dataset.py first")
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_name = args.obj
    wall_json = out_dir / "wall.json"

    # Training -----------------------------------------------------------
    train_seconds = None
    if not args.skip_train:
        from nerfstudio.engine.trainer import Trainer

        cfg = build_config(
            data_dir=data_dir.resolve(),
            output_dir=out_dir.resolve(),
            exp_name=exp_name,
            max_iters=args.max_iters,
            rays_per_batch=args.rays_per_batch,
        )
        cfg.set_timestamp()
        # Write the resolved config to disk so `eval_setup` can find it later.
        cfg.save_config()

        trainer = cfg.setup(local_rank=0, world_size=1)
        trainer.setup()
        t0 = time.time()
        trainer.train()
        train_seconds = time.time() - t0
        print(f"[{args.obj}] training done in {train_seconds:.1f}s")

        # Capture the resolved config path (Trainer wrote `config.yml`).
        run_dir = Path(cfg.get_base_dir())
        load_config_path = run_dir / "config.yml"
        (out_dir / "latest_run.txt").write_text(str(load_config_path))
    else:
        latest = out_dir / "latest_run.txt"
        if not latest.exists():
            raise SystemExit("--skip_train passed but no latest_run.txt found")
        load_config_path = Path(latest.read_text().strip())

    # Export ------------------------------------------------------------
    export_seconds = None
    out_mesh = out_dir / "sdf_mesh.ply"
    if not args.skip_export:
        # NB: resolution must be multiple of 512 per nerfstudio's
        # generate_mesh_with_multires_marching_cubes contract.
        res = max(512, (int(args.resolution) // 512) * 512)
        t0 = time.time()
        export_marching_cubes(
            load_config_path,
            out_mesh,
            resolution=res,
            aabb_min=(-args.aabb_half,) * 3,
            aabb_max=(args.aabb_half,) * 3,
        )
        export_seconds = time.time() - t0
        print(f"[{args.obj}] exported {out_mesh} in {export_seconds:.1f}s (res={res})")

    # Persist wall-time info.
    info = {"obj": args.obj, "max_iters": args.max_iters,
            "rays_per_batch": args.rays_per_batch,
            "train_seconds": train_seconds,
            "export_seconds": export_seconds,
            "resolution": args.resolution,
            "out_mesh": str(out_mesh)}
    wall_json.write_text(json.dumps(info, indent=2))
    print(f"[{args.obj}] wall info -> {wall_json}")


if __name__ == "__main__":
    main()
