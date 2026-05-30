"""DDF forward-pass benchmark using tinycudann (CUDA-fused hash grid + MLP).

Builds a DDF with parameters matching our pure-PyTorch hashgrid DDF
(16 levels x 2 features, base_res 16, growth 1.5, log2 table 19, 2x64 MLP,
4 sinusoidal direction frequencies), random init (timing is weights-independent),
and times the forward pass on the same N_rays sweep used in bench_ddf_vs_bvh.json.

Run on A100. Outputs /tmp/bench_tcnn.json (us per ray, sweep 1K..1M).
"""
from __future__ import annotations

import json
import time
import numpy as np
import torch
import tinycudann as tcnn

assert torch.cuda.is_available()
DEVICE = "cuda:0"

N_RAYS = [1_024, 10_240, 102_400, 1_048_576]
REPS = 10
WARMUPS = 3
BBOX_HALF = 1.2


class TcnnDDF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Hash-grid position encoding (matches src/ddf_hashgrid.py config).
        self.pos = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Grid",
                "type": "Hash",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.5,
            },
        )
        # Sinusoidal direction encoding (4 frequencies -> 3*2*4 = 24 dims).
        self.dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": 4,
            },
        )
        # 2-hidden-layer 64-wide MLP, fused. Output: (t, vis).
        self.mlp = tcnn.Network(
            n_input_dims=self.pos.n_output_dims + self.dir.n_output_dims,
            n_output_dims=2,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

    def forward(self, x, d):
        # Map x from [-bbox, bbox] to [0, 1] for the hash grid.
        x01 = (x + BBOX_HALF) / (2 * BBOX_HALF)
        # Map d from [-1, 1] to [0, 1] for the periodic encoding.
        d01 = d * 0.5 + 0.5
        feat = torch.cat([self.pos(x01), self.dir(d01)], dim=-1)
        out = self.mlp(feat)
        return out[..., 0], out[..., 1]


def time_forward(model, N, reps):
    x = (torch.rand(N, 3, device=DEVICE) * 2 - 1) * 0.5
    d = torch.nn.functional.normalize(torch.randn(N, 3, device=DEVICE), dim=-1)
    for _ in range(WARMUPS):
        with torch.no_grad():
            _ = model(x, d)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x, d)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    print(f"torch {torch.__version__} | tinycudann present | device {torch.cuda.get_device_name(0)}", flush=True)
    model = TcnnDDF().to(DEVICE)
    # Try float16 mode and float32 mode? FullyFusedMLP runs in fp16 by default.
    # Use bf16 autocast for the surrounding ops to match the pytorch baseline.

    out = {
        "n_rays_sweep": N_RAYS,
        "ddf_tcnn": {
            "us_per_ray": {},
            "platform": f"{torch.cuda.get_device_name(0)} | tinycudann FullyFusedMLP + Grid:Hash | torch {torch.__version__}",
            "reps": REPS,
            "warmups": WARMUPS,
            "config": {
                "n_levels": 16, "n_features_per_level": 2,
                "log2_hashmap_size": 19, "base_resolution": 16,
                "per_level_scale": 1.5, "mlp_width": 64, "mlp_layers": 2,
                "dir_freqs": 4, "bbox_half": BBOX_HALF,
            },
        },
    }

    for Nr in N_RAYS:
        t_med = time_forward(model, Nr, REPS)
        us = t_med * 1e6 / Nr
        out["ddf_tcnn"]["us_per_ray"][str(Nr)] = us
        print(f"  N_rays={Nr:>9,}: median {t_med*1e3:8.3f} ms  =>  {us:8.4f} us/ray", flush=True)

    with open("/tmp/bench_tcnn.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote /tmp/bench_tcnn.json")


if __name__ == "__main__":
    main()
