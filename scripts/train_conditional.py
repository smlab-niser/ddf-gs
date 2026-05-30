"""Train a single conditional DDF across many objects (auto-decoder style).

For each step:
  1. ``MultiObjectSupervisor.sample(batch)`` picks a random training object
     and returns a (origins, dirs, t_gt, hit_gt, obj_idx) tuple.
  2. We look up ``z = embedding(obj_idx)``, run the FiLM-conditioned DDF
     forward, and compute the same (L1-dist + BCE-vis) loss as the per-object
     baseline.
  3. Backprop through both the MLP and the embedding -> the per-object
     latent specialises to that object while the MLP learns a shared geometry
     decoder.

The script also accepts a ``--pilot N`` flag for sanity checks: trains on
the first N objects of the manifest, useful to confirm the pipeline before
spending compute on the full 29.

CLI:
    python scripts/train_conditional.py --pilot 2 --steps 2000
    python scripts/train_conditional.py --steps 60000 --out_dir runs/ddf_cond
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _pin_cuda():
    for i, a in enumerate(sys.argv):
        if a == "--cuda_device" and i + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            return
        if a.startswith("--cuda_device="):
            os.environ["CUDA_VISIBLE_DEVICES"] = a.split("=", 1)[1]
            return


_pin_cuda()

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.conditional_supervisor import MultiObjectSupervisor
from src.ddf_conditional import DDFCond


def load_gso_paths(pilot: int | None = None) -> tuple[list[str], list[str]]:
    """Return (short_ids, gs_paths) for the GSO objects with a gaussians.pt."""
    with open("runs/obj_manifest.json") as f:
        gso = json.load(f)
    short_ids = []
    paths = []
    for short in gso:
        gs_path = Path(f"runs/{short}/gaussians.pt")
        if gs_path.exists():
            short_ids.append(short)
            paths.append(str(gs_path))
    if pilot is not None:
        short_ids = short_ids[:pilot]
        paths = paths[:pilot]
    return short_ids, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="runs/ddf_cond")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr_mlp", type=float, default=5.0e-4)
    ap.add_argument("--lr_z", type=float, default=1.0e-3)
    ap.add_argument("--z_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--pos_freqs", type=int, default=10)
    ap.add_argument("--dir_freqs", type=int, default=4)
    ap.add_argument("--lambda_vis", type=float, default=0.1)
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--surface_n_ratio", type=float, default=None)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--pilot", type=int, default=None,
                    help="Use only the first N GSO objects (for quick sanity).")
    ap.add_argument("--cuda_device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    short_ids, gs_paths = load_gso_paths(pilot=args.pilot)
    print(f"loaded {len(short_ids)} objects: {short_ids}")

    # Build supervisor (lazily loads all GS into GPU memory).
    sup = MultiObjectSupervisor(
        gs_paths=gs_paths,
        image_size=args.image_size,
        device=device,
        surface_n_ratio=args.surface_n_ratio,
        seed=args.seed,
    )
    N = sup.num_objects

    # Build model + per-object embedding.
    model = DDFCond(
        pos_freqs=args.pos_freqs,
        dir_freqs=args.dir_freqs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        z_dim=args.z_dim,
    ).to(device)
    z_table = nn.Embedding(N, args.z_dim).to(device)
    nn.init.normal_(z_table.weight, std=0.01)  # DeepSDF init.

    n_mlp = sum(p.numel() for p in model.parameters())
    n_z = sum(p.numel() for p in z_table.parameters())
    print(f"params: MLP={n_mlp:,}  z-table={n_z:,}  total={n_mlp + n_z:,}")

    opt = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": args.lr_mlp},
            {"params": z_table.parameters(), "lr": args.lr_z},
        ]
    )

    # Save short_ids and config first so partial runs are reusable.
    with (out_dir / "objects.json").open("w") as f:
        json.dump(short_ids, f, indent=2)
    with (out_dir / "config.json").open("w") as f:
        cfg_dump = vars(args).copy()
        json.dump(cfg_dump, f, indent=2)

    log_lines = []
    t_start = time.time()
    for step in range(1, args.steps + 1):
        origins, dirs, t_gt, hit_gt, obj_idx = sup.sample(args.batch_size)
        z = z_table(torch.tensor(obj_idx, device=device)).unsqueeze(0)  # (1, z_dim)

        t_pred, vis_logit = model(origins, dirs, z)
        vis_target = hit_gt.float()
        loss_vis = F.binary_cross_entropy_with_logits(vis_logit, vis_target)
        if hit_gt.any():
            loss_dist = F.l1_loss(t_pred[hit_gt], t_gt[hit_gt])
        else:
            loss_dist = torch.zeros((), device=device)
        loss = loss_dist + args.lambda_vis * loss_vis

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                acc = ((vis_logit > 0) == hit_gt).float().mean().item()
            elapsed = time.time() - t_start
            msg = (
                f"step {step:6d} | obj {short_ids[obj_idx]:>12s} | "
                f"loss {loss.item():.4f} | dist {loss_dist.item():.4f} | "
                f"vis {loss_vis.item():.4f} | acc {acc:.3f} | "
                f"elapsed {elapsed:.1f}s"
            )
            print(msg, flush=True)
            log_lines.append(msg)

        if step % args.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "z_table": z_table.state_dict(),
                    "step": step,
                    "objects": short_ids,
                    "args": vars(args),
                },
                out_dir / f"ddf_cond_{step:06d}.pt",
            )

    torch.save(
        {
            "model": model.state_dict(),
            "z_table": z_table.state_dict(),
            "step": args.steps,
            "objects": short_ids,
            "args": vars(args),
        },
        out_dir / "ddf_cond_final.pt",
    )
    (out_dir / "train.log").write_text("\n".join(log_lines))
    elapsed = time.time() - t_start
    print(f"done. wall {elapsed:.1f}s. saved -> {out_dir/'ddf_cond_final.pt'}")


if __name__ == "__main__":
    main()
