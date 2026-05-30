"""DDF distillation training loop. Supervisor selected via cfg['supervisor']."""

import argparse
import os
import sys
from pathlib import Path


# Optional --cuda_device pinning. Must happen BEFORE ``import torch`` so the
# device is selected before CUDA context init. argv is left intact for the
# regular argparse below.
def _maybe_pin_cuda():
    for i, a in enumerate(sys.argv):
        if a == "--cuda_device" and i + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            return
        if a.startswith("--cuda_device="):
            os.environ["CUDA_VISIBLE_DEVICES"] = a.split("=", 1)[1]
            return


_maybe_pin_cuda()

import torch
import torch.nn.functional as F
import yaml

from .ddf_model import DDF
from .ddf_hashgrid import DDFHashGrid
from .ddf_mixture import DDFMixture
from .ddf_kmix2 import DDFKMix2
from .ddf_photometric import DDFHashGridPhotometric
from .ddf_photorender import DDFHashGridPhotoRender
from .ddf_color import DDFHashGridColor
from .gs_supervisor import GSSupervisor
from .photometric_supervisor import PhotometricSupervisor
from .photo_render_supervisor import PhotoRenderSupervisor
from .neus_supervisor import NeuSSupervisor
from .cached_supervisor import CachedSupervisor
from .gtmesh_supervisor import GTMeshSupervisor
from .sphere_supervisor import ray_sphere_intersect, sample_rays


def build_model(cfg: dict):
    """Dispatch on cfg['model']['type'] (default: baseline DDF)."""
    m_cfg = dict(cfg["model"])
    kind = m_cfg.pop("type", "ddf").lower()
    if kind in ("ddf", "baseline"):
        return DDF(
            pos_freqs=m_cfg["pos_freqs"],
            dir_freqs=m_cfg["dir_freqs"],
            hidden_dim=m_cfg["hidden_dim"],
            num_layers=m_cfg["num_layers"],
        )
    if kind == "hashgrid":
        return DDFHashGrid(
            dir_freqs=m_cfg.get("dir_freqs", 4),
            hidden_dim=m_cfg.get("hidden_dim", 64),
            num_layers=m_cfg.get("num_layers", 2),
            n_levels=m_cfg.get("n_levels", 16),
            feat_dim=m_cfg.get("feat_dim", 2),
            log2_table_size=m_cfg.get("log2_table_size", 19),
            base_res=m_cfg.get("base_res", 16),
            growth=m_cfg.get("growth", 1.5),
            bbox_half=m_cfg.get("bbox_half", 1.2),
        )
    if kind == "mixture":
        return DDFMixture(
            K=m_cfg.get("K", 2),
            dir_freqs=m_cfg.get("dir_freqs", 4),
            hidden_dim=m_cfg.get("hidden_dim", 64),
            num_layers=m_cfg.get("num_layers", 2),
            n_levels=m_cfg.get("n_levels", 16),
            feat_dim=m_cfg.get("feat_dim", 2),
            log2_table_size=m_cfg.get("log2_table_size", 19),
            base_res=m_cfg.get("base_res", 16),
            growth=m_cfg.get("growth", 1.5),
            bbox_half=m_cfg.get("bbox_half", 1.2),
        )
    if kind == "kmix2":
        return DDFKMix2(
            dir_freqs=m_cfg.get("dir_freqs", 4),
            hidden_dim=m_cfg.get("hidden_dim", 64),
            num_layers=m_cfg.get("num_layers", 2),
            n_levels=m_cfg.get("n_levels", 16),
            feat_dim=m_cfg.get("feat_dim", 2),
            log2_table_size=m_cfg.get("log2_table_size", 19),
            base_res=m_cfg.get("base_res", 16),
            growth=m_cfg.get("growth", 1.5),
            bbox_half=m_cfg.get("bbox_half", 1.2),
        )
    if kind == "hashgrid_photometric":
        return DDFHashGridPhotometric(
            dir_freqs=m_cfg.get("dir_freqs", 4),
            hidden_dim=m_cfg.get("hidden_dim", 64),
            num_layers=m_cfg.get("num_layers", 2),
            n_levels=m_cfg.get("n_levels", 16),
            feat_dim=m_cfg.get("feat_dim", 2),
            log2_table_size=m_cfg.get("log2_table_size", 19),
            base_res=m_cfg.get("base_res", 16),
            growth=m_cfg.get("growth", 1.5),
            bbox_half=m_cfg.get("bbox_half", 1.2),
        )
    if kind == "hashgrid_photorender":
        return DDFHashGridPhotoRender(
            dir_freqs=m_cfg.get("dir_freqs", 4),
            hidden_dim=m_cfg.get("hidden_dim", 64),
            num_layers=m_cfg.get("num_layers", 2),
            n_levels=m_cfg.get("n_levels", 16),
            feat_dim=m_cfg.get("feat_dim", 2),
            log2_table_size=m_cfg.get("log2_table_size", 19),
            base_res=m_cfg.get("base_res", 16),
            growth=m_cfg.get("growth", 1.5),
            bbox_half=m_cfg.get("bbox_half", 1.2),
            beta_init=m_cfg.get("beta_init", 10.0),
        )
    if kind == "hashgrid_color":
        return DDFHashGridColor(
            dir_freqs=m_cfg.get("dir_freqs", 4),
            hidden_dim=m_cfg.get("hidden_dim", 64),
            num_layers=m_cfg.get("num_layers", 2),
            color_hidden_dim=m_cfg.get("color_hidden_dim", 64),
            color_num_layers=m_cfg.get("color_num_layers", 3),
            n_levels=m_cfg.get("n_levels", 16),
            feat_dim=m_cfg.get("feat_dim", 2),
            log2_table_size=m_cfg.get("log2_table_size", 19),
            base_res=m_cfg.get("base_res", 16),
            growth=m_cfg.get("growth", 1.5),
            bbox_half=m_cfg.get("bbox_half", 1.2),
        )
    raise ValueError(f"unknown model type: {kind}")


class SphereSupervisor:
    """Wraps the Stage-0 analytical sphere sampler in the supervisor interface."""

    def __init__(self, bbox_half: float = 1.5, sphere_radius: float = 1.0, device: str = "cuda"):
        self.bbox_half = bbox_half
        self.sphere_radius = sphere_radius
        self.device = device

    def sample(self, batch_size: int):
        origins, dirs = sample_rays(batch_size, bbox_half=self.bbox_half, device=self.device)
        t_gt, hit_gt = ray_sphere_intersect(origins, dirs, radius=self.sphere_radius)
        return origins, dirs, t_gt, hit_gt


def build_supervisor(cfg: dict):
    device = cfg.get("device", "cuda")
    sup_cfg = cfg.get("supervisor")
    # Back-compat: old Stage-0 configs put bbox_half/sphere_radius at top level.
    if sup_cfg is None:
        sup_cfg = {
            "type": "sphere",
            "bbox_half": cfg.get("bbox_half", 1.5),
            "sphere_radius": cfg.get("sphere_radius", 1.0),
        }

    kind = sup_cfg["type"]
    if kind == "sphere":
        return SphereSupervisor(
            bbox_half=sup_cfg.get("bbox_half", 1.5),
            sphere_radius=sup_cfg.get("sphere_radius", 1.0),
            device=device,
        )
    if kind == "gs":
        return GSSupervisor(
            gs_path=sup_cfg["gs_path"],
            image_size=sup_cfg.get("image_size", 64),
            device=device,
            surface_n_ratio=sup_cfg.get("surface_n_ratio", None),
            surface_chunk=sup_cfg.get("surface_chunk", None),
            march_ratio=sup_cfg.get("march_ratio", 0.0),
        )
    if kind == "gs_photometric":
        return PhotometricSupervisor(
            gs_path=sup_cfg["gs_path"],
            image_size=sup_cfg.get("image_size", 64),
            device=device,
            surface_n_ratio=sup_cfg.get("surface_n_ratio", None),
            surface_chunk=sup_cfg.get("surface_chunk", None),
        )
    if kind == "gs_photo_render":
        return PhotoRenderSupervisor(
            gs_path=sup_cfg["gs_path"],
            views_dir=sup_cfg["views_dir"],
            image_size=sup_cfg.get("image_size", 64),
            device=device,
            surface_n_ratio=sup_cfg.get("surface_n_ratio", None),
            surface_chunk=sup_cfg.get("surface_chunk", None),
            march_ratio=sup_cfg.get("march_ratio", 0.0),
            bbox_half=sup_cfg.get("bbox_half", 1.2),
            n_rgb_rays=sup_cfg.get("n_rgb_rays", 1024),
        )
    if kind == "cached":
        return CachedSupervisor(
            cache_path=sup_cfg["cache_path"],
            device=device,
            march_ratio=sup_cfg.get("march_ratio", 0.5),
        )
    if kind == "gtmesh":
        import numpy as np
        gs = torch.load(sup_cfg["gs_path"], map_location="cpu", weights_only=False)
        return GTMeshSupervisor(
            gt_mesh_path=sup_cfg["gt_mesh_path"],
            mesh_center=gs["mesh_center"].numpy().astype(np.float32),
            mesh_scale=float(gs["mesh_scale"].item()),
            device=device,
            march_ratio=sup_cfg.get("march_ratio", 0.5),
            surface_ratio=sup_cfg.get("surface_ratio", 0.5),
            gso_rotate=sup_cfg.get("gso_rotate", False),
            frustum_ratio=sup_cfg.get("frustum_ratio", 0.0),
            image_size=sup_cfg.get("image_size", 128),
        )
    if kind == "neus":
        return NeuSSupervisor(
            neus_config_path=sup_cfg["neus_config_path"],
            device=device,
            image_size=sup_cfg.get("image_size", 128),
            bbox_half=sup_cfg.get("bbox_half", 1.2),
            march_ratio=sup_cfg.get("march_ratio", 0.5),
            surface_ratio=sup_cfg.get("surface_ratio", 0.3),
            st_max_iter=sup_cfg.get("st_max_iter", 64),
            st_eps=sup_cfg.get("st_eps", 1e-3),
            st_t_far=sup_cfg.get("st_t_far", 5.0),
        )
    raise ValueError(f"unknown supervisor type: {kind}")


def _clamp_l1(pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
    """23ddf-style clamp loss: |clamp(pred, delta) - clamp(target, delta)|.

    Narrows the band the network must fit and yields sharper iso surfaces:
    values above ``delta`` are saturated, so gradient pressure is concentrated
    near the surface (|t - t_gt| < delta zone). Used in Behera & Mishra 2023
    Eq. 8 with delta=0.05.
    """
    pc = pred.clamp(max=delta)
    tc = target.clamp(max=delta)
    return (pc - tc).abs().mean()


def _volume_render(
    model,
    origins: torch.Tensor,    # (R, 3)
    dirs: torch.Tensor,       # (R, 3)
    t_near: torch.Tensor,     # (R,)
    t_far: torch.Tensor,      # (R,)
    n_samples: int,
    bg_color: torch.Tensor,   # (3,)
    jitter: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NeuS-style volume rendering composited from a DDF + RGB head.

    Per ray, sample ``n_samples`` points uniformly (with optional stratified
    jitter) in [t_near, t_far]. Query the DDF distance t_pred and RGB at each.
    Convert distance to density via ``density = beta * exp(-beta * t_pred**2)``
    (an unnormalised Gaussian-bell centred at t_pred=0, i.e. on the surface);
    composite via standard front-to-back alpha = 1 - exp(-density * delta_t).

    Composites a constant ``bg_color`` background to match the GT renders'
    white background convention.

    Returns ``(rgb_rendered (R,3), depth (R,), acc (R,))``.
    """
    R = origins.shape[0]
    device = origins.device

    # Per-ray sample positions along the box-clipped interval.
    # Rays that miss the box have t_far < t_near; replace those with a degenerate
    # zero-length interval so they composite to background.
    eps = 1e-4
    valid = (t_far > t_near + eps)
    t_near_safe = t_near
    t_far_safe = torch.maximum(t_far, t_near + eps)
    # Stratified sample bins.
    u = torch.linspace(0.0, 1.0, n_samples + 1, device=device)        # (N+1,)
    bin_lo = u[:-1].view(1, n_samples)                                 # (1, N)
    bin_hi = u[1:].view(1, n_samples)                                  # (1, N)
    if jitter:
        rand = torch.rand(R, n_samples, device=device)
    else:
        rand = torch.full((R, n_samples), 0.5, device=device)
    u_samples = bin_lo + (bin_hi - bin_lo) * rand                      # (R, N) in [0,1]
    t_samples = t_near_safe.unsqueeze(-1) + (
        t_far_safe - t_near_safe
    ).unsqueeze(-1) * u_samples                                        # (R, N)

    # Sample positions and per-sample direction (broadcast).
    pos = origins.unsqueeze(1) + t_samples.unsqueeze(-1) * dirs.unsqueeze(1)  # (R,N,3)
    dirs_per = dirs.unsqueeze(1).expand(-1, n_samples, -1)                   # (R,N,3)

    flat_pos = pos.reshape(-1, 3)
    flat_dir = dirs_per.reshape(-1, 3)
    # Forward through model. DDFHashGridPhotoRender returns (t, vis_logit, rgb).
    t_pred_flat, _vis_logit_flat, rgb_flat = model(flat_pos, flat_dir)
    t_pred = t_pred_flat.view(R, n_samples)             # predicted DDF at each sample
    rgb = rgb_flat.view(R, n_samples, 3)

    # Density: a Gaussian-bell on t_pred centred at 0 (surface). beta is the
    # learnable sharpness.
    beta = model.beta  # () > 0
    density = beta * torch.exp(-beta * t_pred * t_pred)  # (R, N)

    # Per-sample step length.
    # Use the spacing between sampled points (last gap = same as previous).
    delta_t = torch.diff(t_samples, dim=-1)                                  # (R, N-1)
    last_delta = delta_t[..., -1:].clamp_min(0.0)
    delta_t = torch.cat([delta_t, last_delta], dim=-1)                        # (R, N)
    # Zero out contributions on rays that don't actually intersect the box.
    delta_t = delta_t * valid.unsqueeze(-1).float()

    alpha = 1.0 - torch.exp(-density * delta_t)                              # (R, N) in [0,1]
    # Transmittance: T_i = prod_{k<i} (1 - alpha_k)
    one_minus_alpha = 1.0 - alpha + 1e-10
    T = torch.cumprod(
        torch.cat([torch.ones(R, 1, device=device), one_minus_alpha[:, :-1]], dim=-1),
        dim=-1,
    )
    weights = alpha * T                                                       # (R, N)

    rgb_rendered = (weights.unsqueeze(-1) * rgb).sum(dim=1)                   # (R, 3)
    depth = (weights * t_samples).sum(dim=1)                                  # (R,)
    acc = weights.sum(dim=1)                                                  # (R,)

    rgb_rendered = rgb_rendered + (1.0 - acc).unsqueeze(-1) * bg_color.view(1, 3)
    return rgb_rendered, depth, acc


def train(cfg: dict):
    device = cfg.get("device", "cuda")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {type(model).__name__}  params: {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # Optional cosine LR schedule (used for hi-res / large-batch runs).
    lr_schedule = cfg.get("lr_schedule")
    if lr_schedule == "cosine":
        eta_min = cfg["lr"] * cfg.get("lr_min_ratio", 0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["steps"], eta_min=eta_min)
        print(f"using cosine LR schedule: {cfg['lr']} -> {eta_min}")
    else:
        scheduler = None

    supervisor = build_supervisor(cfg)

    log_every = cfg.get("log_every", 200)
    save_every = cfg.get("save_every", 2000)
    steps = cfg["steps"]
    batch = cfg["batch_size"]
    lam_vis = cfg["lambda_vis"]
    lam_rgb = float(cfg.get("lambda_rgb", 0.0))

    # Photometric path: model returns (t, vis, rgb) and supervisor yields rgb_gt.
    is_photometric = isinstance(model, DDFHashGridPhotometric)
    # Mixture path: model exposes .forward_all -> (d_all, w_all, vis_logit).
    is_mixture = isinstance(model, DDFMixture)
    # K=2 mixture w/ per-mode visibility: model returns (d1, d2, xi1, xi2).
    is_kmix2 = isinstance(model, DDFKMix2)
    # Photo-volume-rendering path: NeuS-style supervision via N=32-64 samples/ray.
    is_photorender = isinstance(model, DDFHashGridPhotoRender)
    # Color DDF: separate geo/color branches, sphere-trace + direct RGB regression.
    is_color = isinstance(model, DDFHashGridColor)

    # Color training config (only used when is_color).
    color_cfg = cfg.get("color", {}) or {}
    color_views_dir = color_cfg.get("views_dir")
    color_n_rays = int(color_cfg.get("n_rays", 1024))
    color_warmup = int(color_cfg.get("warmup_steps", 5000))
    color_images = None
    color_cameras = None
    if is_color and color_views_dir:
        import numpy as np
        from PIL import Image as _PILImage
        _vdir = Path(color_views_dir)
        _cam = np.load(_vdir / "cameras.npz")
        color_cameras = {
            "c2w": torch.from_numpy(_cam["c2w"]).float().to(device),
            "K": torch.from_numpy(_cam["K"]).float().to(device),
            "image_size": int(_cam["image_size"]),
        }
        _imgs = sorted(_vdir.glob("rgb_*.png"))
        color_images = []
        for _ip in _imgs:
            _im = np.array(_PILImage.open(_ip).convert("RGB"), dtype=np.float32) / 255.0
            color_images.append(torch.from_numpy(_im).to(device))
        color_images = torch.stack(color_images)
        print(f"color training: {len(color_images)} views, {color_cameras['image_size']}px, "
              f"{color_n_rays} rays/step, warmup {color_warmup}")

    # Volume-rendering config (only used when is_photorender).
    vr_cfg = cfg.get("volume_render", {}) or {}
    vr_n_samples = int(vr_cfg.get("n_samples", 32))
    vr_n_rays = int(vr_cfg.get("n_rays", 1024))
    vr_jitter = bool(vr_cfg.get("jitter", True))
    bg_color = torch.tensor(vr_cfg.get("bg_color", [1.0, 1.0, 1.0]),
                            device=device, dtype=torch.float32)

    # Iso-aware loss config (PDDF-style mixture; 23ddf-style clamp).
    loss_cfg = cfg.get("loss", {}) or {}
    # Default: clamp ON when mixture is enabled (Stage 14 recipe).
    use_clamp = bool(loss_cfg.get("clamp", is_mixture))
    delta = float(loss_cfg.get("delta", 0.05))
    gamma_d = float(loss_cfg.get("gamma_d", 5.0 if is_mixture else 1.0))
    if is_mixture or "gamma_xi" in loss_cfg:
        gamma_xi = float(loss_cfg.get("gamma_xi", 1.0))
    else:
        # Back-compat: legacy path keeps lambda_vis as the visibility weight.
        gamma_xi = lam_vis
    # Eikonal-along-ray loss: enforce d/dx DDF(x,d) . d + 1 = 0 on hit rays
    # (PDDF Aumentado-Armstrong 2022 Eq. for L_E,d). Off by default; enable
    # via cfg.loss.gamma_eikonal > 0. Mixture path is not supported (we don't
    # have a smooth path through argmax over K components).
    gamma_eikonal = float(loss_cfg.get("gamma_eikonal", 0.0))
    if is_mixture and gamma_eikonal > 0:
        print("WARNING: gamma_eikonal>0 not supported for mixture; ignoring.")
        gamma_eikonal = 0.0

    extra_kmix2 = ""
    if is_kmix2:
        extra_kmix2 = (
            f"  gamma_xi2={float(loss_cfg.get('gamma_xi2', 0.0))}"
            f"  gamma_xi2_hit={float(loss_cfg.get('gamma_xi2_hit', 0.0))}"
            f"  gamma_order={float(loss_cfg.get('gamma_order', 0.0))}"
        )
    print(
        f"loss config: mixture={is_mixture}  kmix2={is_kmix2}  clamp={use_clamp}  "
        f"delta={delta}  gamma_d={gamma_d}  gamma_xi={gamma_xi}  "
        f"gamma_eikonal={gamma_eikonal}{extra_kmix2}"
    )

    for step in range(1, steps + 1):
        sample = supervisor.sample(batch)
        rgb_origins = rgb_dirs = rgb_gt_vr = t_near_vr = t_far_vr = None
        if is_photorender and len(sample) == 9:
            (origins, dirs, t_gt, hit_gt,
             rgb_origins, rgb_dirs, rgb_gt_vr, t_near_vr, t_far_vr) = sample
            rgb_gt = None
        elif is_photometric and len(sample) == 5:
            origins, dirs, t_gt, hit_gt, rgb_gt = sample
        else:
            origins, dirs, t_gt, hit_gt = sample
            rgb_gt = None

        d_all = w_all = None
        # K=2-mixture-with-per-mode-visibility extras (logged only).
        d1_pred = d2_pred = xi1_logit = xi2_logit = None
        if is_photorender:
            t_pred, vis_logit, _rgb_unused = model(origins, dirs)
            rgb_pred = None
        elif is_photometric:
            t_pred, vis_logit, rgb_pred = model(origins, dirs)
        elif is_mixture:
            d_all, w_all, vis_logit = model.forward_all(origins, dirs)
            idx = w_all.argmax(dim=-1, keepdim=True)
            t_pred = d_all.gather(-1, idx).squeeze(-1)
            rgb_pred = None
        elif is_kmix2:
            d1_pred, d2_pred, xi1_logit, xi2_logit = model(origins, dirs)
            # Enforce d_1 < d_2 by swapping pairwise where it isn't.
            swap = d1_pred > d2_pred
            d1_sorted = torch.where(swap, d2_pred, d1_pred)
            d2_sorted = torch.where(swap, d1_pred, d2_pred)
            xi1_sorted_logit = torch.where(swap, xi2_logit, xi1_logit)
            xi2_sorted_logit = torch.where(swap, xi1_logit, xi2_logit)
            # Primary mode (near surface) supervised by t_gt.
            t_pred = d1_sorted
            vis_logit = xi1_sorted_logit
            rgb_pred = None
        elif is_color:
            t_pred, vis_logit = model.forward_geom(origins, dirs)
            rgb_pred = None
        else:
            t_pred, vis_logit = model(origins, dirs)
            rgb_pred = None

        vis_target = hit_gt.float()
        loss_vis = F.binary_cross_entropy_with_logits(vis_logit, vis_target)
        if hit_gt.any():
            pred_hit = t_pred[hit_gt]
            tgt_hit = t_gt[hit_gt]
            if use_clamp:
                loss_dist = _clamp_l1(pred_hit, tgt_hit, delta)
            else:
                loss_dist = F.l1_loss(pred_hit, tgt_hit)
        else:
            loss_dist = torch.zeros((), device=device)

        if is_photometric and rgb_gt is not None and hit_gt.any():
            loss_rgb = F.l1_loss(rgb_pred[hit_gt], rgb_gt[hit_gt])
        else:
            loss_rgb = torch.zeros((), device=device)

        # NeuS-style volume rendering loss for the photo-render variant.
        loss_vr = torch.zeros((), device=device)
        if is_photorender and rgb_gt_vr is not None:
            rgb_rendered, _depth_vr, _acc_vr = _volume_render(
                model, rgb_origins, rgb_dirs, t_near_vr, t_far_vr,
                n_samples=vr_n_samples, bg_color=bg_color, jitter=vr_jitter,
            )
            loss_vr = F.l1_loss(rgb_rendered, rgb_gt_vr)

        # K=2 per-mode-visibility extras:
        #   - L_xi2_miss : on MISS rays both modes should be invisible, so
        #     train xi_2 to 0 on miss (xi_1 is already by primary BCE above).
        #   - L_xi2_hit  : on HIT rays, optional weak positive signal pulling
        #     xi_2 logit toward xi_1's logit (detached). Without this xi_2
        #     collapses to 0 everywhere because nothing pushes it up.
        #   - L_d_order  : soft margin enforcing d_2 >= d_1 + delta on the
        #     pre-swap outputs (the swap is non-differentiable so the soft
        #     loss gives gradient to both heads).
        loss_xi2 = torch.zeros((), device=device)
        loss_xi2_hit = torch.zeros((), device=device)
        loss_d_order = torch.zeros((), device=device)
        gamma_xi2 = float(loss_cfg.get("gamma_xi2", 0.0)) if is_kmix2 else 0.0
        gamma_xi2_hit = float(loss_cfg.get("gamma_xi2_hit", 0.0)) if is_kmix2 else 0.0
        gamma_order = float(loss_cfg.get("gamma_order", 0.0)) if is_kmix2 else 0.0
        if is_kmix2:
            miss = ~hit_gt
            if gamma_xi2 > 0 and miss.any():
                loss_xi2 = F.binary_cross_entropy_with_logits(
                    xi2_sorted_logit[miss],
                    torch.zeros_like(xi2_sorted_logit[miss]),
                )
            if gamma_xi2_hit > 0 and hit_gt.any():
                # MSE between xi_2 logit and xi_1 logit (detached) on hit rays.
                # Encourages xi_2 to track xi_1 (so a second surface is *available*
                # wherever a first is), but doesn't override the trunk's own choice
                # since the detach prevents xi_1 from being pulled toward xi_2.
                loss_xi2_hit = (
                    (xi2_sorted_logit[hit_gt]
                     - xi1_sorted_logit[hit_gt].detach()) ** 2
                ).mean()
            if gamma_order > 0:
                gap = d2_pred - d1_pred  # signed; want >= delta
                loss_d_order = F.relu(delta - gap).mean()

        if is_mixture:
            loss = gamma_d * loss_dist + gamma_xi * loss_vis
        elif is_kmix2:
            loss = (
                loss_dist
                + lam_vis * loss_vis
                + gamma_xi2 * loss_xi2
                + gamma_xi2_hit * loss_xi2_hit
                + gamma_order * loss_d_order
            )
        elif is_photorender:
            loss = loss_dist + lam_vis * loss_vis + lam_rgb * loss_vr
        else:
            loss = loss_dist + lam_vis * loss_vis + lam_rgb * loss_rgb

        # --- Eikonal-along-ray loss ---
        # Analytical identity: for any ray (x, d), DDF(x, d) decreases at exactly
        # rate 1 as we step along d toward the surface, so
        #     grad_x DDF(x,d) . d + 1 = 0
        # We evaluate this on the hit subset (non-hit rays have no defined t).
        loss_eikonal = torch.zeros((), device=device)
        if gamma_eikonal > 0 and (not is_mixture) and (not is_kmix2) and hit_gt.any():
            o_hit = origins[hit_gt].detach().requires_grad_(True)
            d_hit = dirs[hit_gt].detach()
            out_e = model(o_hit, d_hit)
            t_pred_e = out_e[0]
            grad = torch.autograd.grad(
                t_pred_e.sum(), o_hit, create_graph=True
            )[0]
            along_ray = (grad * d_hit).sum(-1)
            loss_eikonal = ((along_ray + 1.0) ** 2).mean()
            loss = loss + gamma_eikonal * loss_eikonal

        # --- Color + silhouette loss from multi-view images ---
        loss_color = torch.zeros((), device=device)
        loss_sil = torch.zeros((), device=device)
        if is_color and color_images is not None and step > color_warmup:
            lam_c = min(1.0, (step - color_warmup) / 5000.0) * lam_rgb
            sil_w = float(color_cfg.get("silhouette_weight", 0.0))
            if lam_c > 0 or sil_w > 0:
                n_views = color_images.shape[0]
                vi = torch.randint(0, n_views, ()).item()
                c2w_v = color_cameras["c2w"][vi]
                K_v = color_cameras["K"]
                isz = color_cameras["image_size"]
                gt_img = color_images[vi]
                fg_mask = (gt_img.sum(-1) < 2.95)

                # Sample fg + bg pixels for silhouette loss
                n_sil = color_n_rays
                fg_idx = fg_mask.nonzero()
                bg_idx = (~fg_mask).nonzero()
                n_fg_sil = min(int(n_sil * 0.7), fg_idx.shape[0])
                n_bg_sil = min(n_sil - n_fg_sil, bg_idx.shape[0])
                sel_fg = fg_idx[torch.randperm(fg_idx.shape[0], device=device)[:n_fg_sil]]
                sel_bg = bg_idx[torch.randperm(bg_idx.shape[0], device=device)[:n_bg_sil]]
                sel_all = torch.cat([sel_fg, sel_bg], dim=0)
                fg_gt_sil = torch.cat([
                    torch.ones(n_fg_sil, device=device),
                    torch.zeros(n_bg_sil, device=device)])

                if sel_all.shape[0] > 0:
                    py = (sel_all[:, 0].float() + 0.5 - K_v[1, 2]) / K_v[1, 1]
                    px = (sel_all[:, 1].float() + 0.5 - K_v[0, 2]) / K_v[0, 0]
                    dirs_cam = torch.stack([px, py, torch.ones_like(px)], dim=-1)
                    dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True)
                    R_v = c2w_v[:3, :3]
                    dirs_w = dirs_cam @ R_v.T
                    origins_w = c2w_v[:3, 3].unsqueeze(0).expand_as(dirs_w)

                    # Silhouette loss: vis_logit at camera origin should match fg mask
                    if sil_w > 0:
                        _, vis_sil = model.forward_geom(origins_w, dirs_w)
                        loss_sil = F.binary_cross_entropy_with_logits(vis_sil, fg_gt_sil)
                        loss = loss + sil_w * loss_sil

                    # Color loss on foreground rays only
                    if lam_c > 0 and n_fg_sil > 0:
                        fg_origins = origins_w[:n_fg_sil]
                        fg_dirs = dirs_w[:n_fg_sil]
                        fg_rgb_gt = gt_img[sel_fg[:, 0], sel_fg[:, 1]]
                        with torch.no_grad():
                            t_st = torch.zeros(n_fg_sil, device=device)
                            alive = torch.ones(n_fg_sil, dtype=torch.bool, device=device)
                            for _k in range(32):
                                if not alive.any():
                                    break
                                _idx = alive.nonzero(as_tuple=True)[0]
                                _x = fg_origins[_idx] + t_st[_idx].unsqueeze(-1) * fg_dirs[_idx]
                                _d, _v = model.forward_geom(_x, fg_dirs[_idx])
                                t_st[_idx] += _d
                                converged = _d < 0.05
                                escaped = t_st[_idx] > 5.0
                                alive[_idx[converged]] = False
                                alive[_idx[escaped]] = False
                            st_hit = (t_st < 5.0) & (~alive)
                        if st_hit.any():
                            x_surf = fg_origins[st_hit] + t_st[st_hit].unsqueeze(-1) * fg_dirs[st_hit]
                            _, _, rgb_pred = model(x_surf, fg_dirs[st_hit])
                            loss_color = F.l1_loss(rgb_pred, fg_rgb_gt[st_hit])
                            loss = loss + lam_c * loss_color

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()

        if step % log_every == 0:
            with torch.no_grad():
                acc = ((vis_logit > 0) == hit_gt).float().mean().item()
                extra = ""
                if is_mixture and hit_gt.any():
                    d_h = d_all[hit_gt]
                    d_spread = (d_h.max(-1).values - d_h.min(-1).values).mean().item()
                    extra = f" | d_spread {d_spread:.3f}"
                if is_kmix2 and hit_gt.any():
                    # d-spread on hit rays (after swap: d2 - d1).
                    d1_h = d1_pred[hit_gt]
                    d2_h = d2_pred[hit_gt]
                    spread = (torch.maximum(d1_h, d2_h)
                              - torch.minimum(d1_h, d2_h)).mean().item()
                    xi2_h = torch.sigmoid(xi2_logit[hit_gt]).mean().item()
                    extra = (
                        f" | d_spread {spread:.3f} | xi2_h {xi2_h:.3f}"
                        f" | xi2L {loss_xi2.item():.3f}"
                        f" | xi2hL {loss_xi2_hit.item():.3f}"
                        f" | ordL {loss_d_order.item():.3f}"
                    )
                if gamma_eikonal > 0:
                    extra = extra + f" | eik {loss_eikonal.item():.4f}"
                if is_color and (loss_color.item() > 0 or loss_sil.item() > 0):
                    extra = extra + f" | col {loss_color.item():.4f} | sil {loss_sil.item():.4f}"
            if is_photorender:
                with torch.no_grad():
                    beta_val = float(model.beta.detach())
                print(
                    f"step {step:6d} | loss {loss.item():.4f} | "
                    f"dist {loss_dist.item():.4f} | vis {loss_vis.item():.4f} | "
                    f"vr {loss_vr.item():.4f} | beta {beta_val:.2f} | "
                    f"acc {acc:.3f}{extra}"
                )
            elif is_photometric:
                print(
                    f"step {step:6d} | loss {loss.item():.4f} | "
                    f"dist {loss_dist.item():.4f} | vis {loss_vis.item():.4f} | "
                    f"rgb {loss_rgb.item():.4f} | acc {acc:.3f}{extra}"
                )
            else:
                print(
                    f"step {step:6d} | loss {loss.item():.4f} | "
                    f"dist {loss_dist.item():.4f} | vis {loss_vis.item():.4f} | "
                    f"acc {acc:.3f}{extra}"
                )

        if step % save_every == 0:
            torch.save(
                {"model": model.state_dict(), "step": step, "cfg": cfg},
                out_dir / f"ddf_{step:06d}.pt",
            )

    torch.save({"model": model.state_dict(), "step": steps, "cfg": cfg}, out_dir / "ddf_final.pt")
    print(f"saved final ckpt to {out_dir / 'ddf_final.pt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/mvp.yaml")
    ap.add_argument("--gs_path", type=str, default=None, help="override supervisor.gs_path")
    ap.add_argument("--out_dir", type=str, default=None, help="override cfg out_dir")
    ap.add_argument("--cuda_device", type=str, default=None,
                    help="GPU index to pin via CUDA_VISIBLE_DEVICES (handled before torch import)")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.gs_path is not None:
        cfg.setdefault("supervisor", {})["gs_path"] = args.gs_path
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    train(cfg)


if __name__ == "__main__":
    main()
