"""Image-driven DDF training via differentiable sphere-tracing.

Stage 2: Learn DDF geometry + color directly from multi-view images.
No GS depth, no NeuS — just images → DDF.

Two-phase training:
  Phase A (0 to phase_b_start): Silhouette loss only. Random rays, predict
    vis_logit, compare to foreground mask. Bootstraps coarse geometry.
  Phase B (phase_b_start to end): Differentiable sphere-trace + color loss.
    Sphere-trace (no_grad) to find surface, one differentiable refinement step,
    predict color, compare to GT pixel.

Losses:
  L_mask:    BCE(vis_logit, fg_gt) — silhouette supervision
  L_color:   L1(predicted_rgb, gt_rgb) at sphere-traced hit points
  L_eikonal: (grad_x DDF(x,d) . d + 1)^2 — ray-directional smoothness
  L_march:   L1(DDF(o+s*d, d), DDF(o, d) - s) — self-consistency along rays
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from .ddf_color import DDFHashGridColor
from .image_supervisor import ImageSupervisor


@torch.no_grad()
def _sphere_trace_batch(model, origins, dirs, max_iter=48, eps=0.05, t_far=5.0):
    """No-grad sphere-trace. Returns (t, hit_mask)."""
    N = origins.shape[0]
    device = origins.device
    t = torch.zeros(N, device=device)
    alive = torch.ones(N, dtype=torch.bool, device=device)
    hit = torch.zeros(N, dtype=torch.bool, device=device)
    for _ in range(max_iter):
        if not alive.any():
            break
        idx = alive.nonzero(as_tuple=True)[0]
        x = origins[idx] + t[idx].unsqueeze(-1) * dirs[idx]
        d, v = model.forward_geom(x, dirs[idx])
        is_hit = d < eps
        if is_hit.any():
            hi = idx[is_hit]
            hit[hi] = True
            alive[hi] = False
        t[idx] += d
        escaped = t[idx] > t_far
        if escaped.any():
            alive[idx[escaped]] = False
    return t, hit


def train(cfg: dict):
    device = cfg.get("device", "cuda")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = DDFHashGridColor(
        dir_freqs=cfg["model"].get("dir_freqs", 4),
        hidden_dim=cfg["model"].get("hidden_dim", 64),
        num_layers=cfg["model"].get("num_layers", 2),
        color_hidden_dim=cfg["model"].get("color_hidden_dim", 64),
        color_num_layers=cfg["model"].get("color_num_layers", 3),
        n_levels=cfg["model"].get("n_levels", 16),
        feat_dim=cfg["model"].get("feat_dim", 2),
        log2_table_size=cfg["model"].get("log2_table_size", 19),
        base_res=cfg["model"].get("base_res", 16),
        growth=cfg["model"].get("growth", 1.5),
        bbox_half=cfg["model"].get("bbox_half", 1.2),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {type(model).__name__}  params: {n_params:,}")

    warmstart = cfg.get("warmstart_ckpt")
    if warmstart and Path(warmstart).exists():
        ws = torch.load(warmstart, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ws["model"], strict=False)
        print(f"warm-start from {warmstart} (missing={len(missing)}, unexpected={len(unexpected)})")

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = None
    if cfg.get("lr_schedule") == "cosine":
        eta_min = cfg["lr"] * cfg.get("lr_min_ratio", 0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg["steps"], eta_min=eta_min)

    sup = ImageSupervisor(
        views_dir=cfg["supervisor"]["views_dir"],
        device=device,
        fg_ratio=cfg["supervisor"].get("fg_ratio", 0.7),
    )

    steps = cfg["steps"]
    batch = cfg["batch_size"]
    log_every = cfg.get("log_every", 500)
    save_every = cfg.get("save_every", 10000)

    lam_mask = float(cfg.get("lambda_mask", 1.0))
    lam_color = float(cfg.get("lambda_color", 1.0))
    lam_eikonal = float(cfg.get("lambda_eikonal", 0.1))
    lam_march = float(cfg.get("lambda_march", 0.5))
    phase_b_start = int(cfg.get("phase_b_start", 5000))
    bbox_half = cfg["model"].get("bbox_half", 1.2)

    print(f"phase A (silhouette): steps 0-{phase_b_start}")
    print(f"phase B (color+geom): steps {phase_b_start}-{steps}")
    print(f"lambdas: mask={lam_mask} color={lam_color} eik={lam_eikonal} march={lam_march}")

    for step in range(1, steps + 1):
        data = sup.sample(batch)
        origins = data["origins"]
        dirs = data["dirs"]
        rgb_gt = data["rgb_gt"]
        fg_gt = data["fg_gt"]

        # --- Silhouette loss (both phases) ---
        # Sample random points along each ray to check visibility
        # Use a coarse probe: evaluate DDF at the camera origin direction
        t_probe, vis_logit_probe = model.forward_geom(origins, dirs)
        loss_mask = F.binary_cross_entropy_with_logits(vis_logit_probe, fg_gt.float())

        # --- March consistency (both phases) ---
        # Self-supervised: for random (o, d), DDF(o + s*d, d) should = DDF(o, d) - s
        loss_march = torch.zeros((), device=device)
        if lam_march > 0:
            n_march = min(512, batch // 4)
            m_idx = torch.randint(0, batch, (n_march,), device=device)
            m_o = origins[m_idx].detach()
            m_d = dirs[m_idx].detach()
            with torch.no_grad():
                m_t0, _ = model.forward_geom(m_o, m_d)
            s = torch.empty(n_march, device=device).uniform_(0.05, 0.5) * m_t0.clamp(max=2.0)
            m_o2 = m_o + s.unsqueeze(-1) * m_d
            m_t1, _ = model.forward_geom(m_o2, m_d)
            loss_march = F.l1_loss(m_t1, (m_t0.detach() - s).clamp(min=0))

        # --- Eikonal along-ray loss (both phases) ---
        loss_eikonal = torch.zeros((), device=device)
        if lam_eikonal > 0:
            n_eik = min(512, batch // 4)
            e_idx = torch.randint(0, batch, (n_eik,), device=device)
            e_o = origins[e_idx].detach().requires_grad_(True)
            e_d = dirs[e_idx].detach()
            e_t, _ = model.forward_geom(e_o, e_d)
            grad = torch.autograd.grad(e_t.sum(), e_o, create_graph=True)[0]
            along_ray = (grad * e_d).sum(-1)
            loss_eikonal = ((along_ray + 1.0) ** 2).mean()

        # --- Phase B: differentiable sphere-trace (Tier 2: last-K diff steps) ---
        loss_color = torch.zeros((), device=device)
        loss_depth_img = torch.zeros((), device=device)
        n_diff_steps = int(cfg.get("n_diff_steps", 3))
        if step >= phase_b_start:
            fg_origins = origins[fg_gt]
            fg_dirs = dirs[fg_gt]
            fg_rgb_gt = rgb_gt[fg_gt]

            if fg_origins.shape[0] > 0:
                # No-grad sphere-trace for first (max_iter - K) steps
                t_st, st_hit = _sphere_trace_batch(
                    model, fg_origins, fg_dirs,
                    max_iter=max(1, 48 - n_diff_steps), eps=0.05, t_far=5.0)

                if st_hit.any():
                    hit_o = fg_origins[st_hit]
                    hit_d = fg_dirs[st_hit]
                    hit_t = t_st[st_hit].detach()
                    hit_rgb_gt = fg_rgb_gt[st_hit]

                    # Last K differentiable steps — gradient flows through
                    # the distance predictions to the DDF geometry head
                    t_diff = hit_t
                    for _k in range(n_diff_steps):
                        x_k = hit_o + t_diff.unsqueeze(-1) * hit_d
                        dist_k, _ = model.forward_geom(x_k, hit_d)
                        t_diff = t_diff + dist_k

                    # Surface point with gradient chain
                    x_surface = hit_o + t_diff.unsqueeze(-1) * hit_d

                    # Color at the differentiable surface point
                    _, _, rgb_pred = model(x_surface, hit_d)
                    loss_color = F.l1_loss(rgb_pred, hit_rgb_gt)

                    # Depth consistency: the final dist_k should be near zero (at surface)
                    loss_depth_img = dist_k.abs().mean()

        # --- Total loss ---
        color_ramp = min(1.0, max(0.0, (step - phase_b_start) / 5000.0)) if step >= phase_b_start else 0.0
        lam_depth_img = float(cfg.get("lambda_depth_img", 0.1))
        loss = (lam_mask * loss_mask
                + lam_color * color_ramp * loss_color
                + lam_eikonal * loss_eikonal
                + lam_march * loss_march
                + lam_depth_img * color_ramp * loss_depth_img)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()

        if step % log_every == 0:
            with torch.no_grad():
                vis_acc = ((vis_logit_probe > 0) == fg_gt).float().mean().item()
            print(
                f"step {step:6d} | loss {loss.item():.4f} | "
                f"mask {loss_mask.item():.4f} | col {loss_color.item():.4f} | "
                f"eik {loss_eikonal.item():.4f} | march {loss_march.item():.4f} | "
                f"d_img {loss_depth_img.item():.4f} | "
                f"vis_acc {vis_acc:.3f} | ramp {color_ramp:.2f}"
            )

        if step % save_every == 0:
            torch.save({"model": model.state_dict(), "step": step, "cfg": cfg},
                       out_dir / f"ddf_{step:06d}.pt")

    torch.save({"model": model.state_dict(), "step": steps, "cfg": cfg},
               out_dir / "ddf_final.pt")
    print(f"saved final ckpt to {out_dir / 'ddf_final.pt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
