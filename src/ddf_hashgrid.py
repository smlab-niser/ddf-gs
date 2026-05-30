"""DDFHashGrid: hash-grid positional encoding variant of DDF.

Uses a multiresolution hash grid (instant-ngp, Müller et al. 2022) for the
3D point encoding, retaining sinusoidal encoding on the direction. The small
MLP trunk + two heads (distance, vis_logit) mirror :class:`DDF`.

The hash grid is implemented in pure PyTorch (no CUDA extension); ~16 levels
of feature grids, 2 features per level (~32-dim output), trilinear interpolation
with a spatial-hash collision lookup on the finer levels. Reference:
https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf §3.

Implementation note: the level loop is vectorised — corner indices for all
levels are computed in a single batched op, then we do exactly 8 ``index_select``
calls (one per cube corner) into a single concatenated embedding table.

API matches :class:`DDF` exactly, so train_ddf.py / stage3_chamfer.py just need
to know which class to instantiate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ddf_model import sinusoidal_encoding


# Three large primes for the spatial-hash function (instant-ngp Eq. 4).
_HASH_PRIMES = (1, 2654435761, 805459861)


class HashGridEncoder(nn.Module):
    """Multiresolution hash-grid encoding (instant-ngp style), pure PyTorch.

    Each of ``n_levels`` levels has its own resolution ``N_l`` (geometric
    progression from ``base_res`` by factor ``b``) and a feature table of
    size ``min(N_l^3, 2^log2_table_size)``. Feature is a trilinear interpolation
    of the 8 corners of the voxel containing ``x``. The per-level features
    (``feat_dim`` each) are concatenated → output dim = ``n_levels * feat_dim``.

    Coordinates are expected in the cube ``[-bbox_half, bbox_half]^3``; they are
    rescaled to ``[0, N_l - 1]`` per level.
    """

    def __init__(
        self,
        n_levels: int = 16,
        feat_dim: int = 2,
        log2_table_size: int = 19,
        base_res: int = 16,
        growth: float = 1.5,
        bbox_half: float = 1.2,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.feat_dim = feat_dim
        self.log2_table_size = log2_table_size
        self.base_res = base_res
        self.growth = growth
        self.bbox_half = bbox_half

        max_T = 1 << log2_table_size
        resolutions = []
        table_sizes = []
        offsets = [0]
        for l in range(n_levels):
            N_l = int(round(base_res * (growth ** l)))
            # Use a dense table whenever it fits in the budget (lossless for
            # coarse levels); hash beyond that.
            T_l = min(N_l ** 3, max_T)
            resolutions.append(N_l)
            table_sizes.append(T_l)
            offsets.append(offsets[-1] + T_l)
        total_entries = offsets[-1]

        # One concatenated parameter buffer (cheaper indexing than a list of
        # per-level params; ~4 MB for 16 levels × 2^19 × 2 fp32 entries).
        self.embeddings = nn.Parameter(torch.empty(total_entries, feat_dim))
        nn.init.uniform_(self.embeddings, -1e-4, 1e-4)

        # Per-level tensors broadcast over (N_points, n_levels). Float for the
        # rescale (will be cast back to int for hashing), int64 for offsets/table
        # sizes.
        self.register_buffer(
            "res_f", torch.tensor(resolutions, dtype=torch.float32),
        )
        self.register_buffer(
            "res_i", torch.tensor(resolutions, dtype=torch.int64),
        )
        self.register_buffer(
            "table_sizes", torch.tensor(table_sizes, dtype=torch.int64),
        )
        self.register_buffer(
            "offsets", torch.tensor(offsets[:-1], dtype=torch.int64),
        )
        # Whether level uses dense (no hash) indexing — boolean per level.
        dense_mask = torch.tensor(
            [T == N ** 3 for T, N in zip(table_sizes, resolutions)],
            dtype=torch.bool,
        )
        self.register_buffer("dense_mask", dense_mask)

        self.out_dim = n_levels * feat_dim

    def _lookup_corner(
        self,
        ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor,
    ) -> torch.Tensor:
        """Indices: (N, L) each. Returns features: (N, L, F).

        Uses per-level dense indexing where the level is small enough to fit a
        dense table; otherwise spatial hash. Both produce table-local indices
        that are then offset into the single concatenated embedding tensor.
        """
        # Hash path (works for all levels; we only override with dense where
        # advantageous).
        ix64 = ix.to(torch.int64)
        iy64 = iy.to(torch.int64)
        iz64 = iz.to(torch.int64)
        hashed = (
            (ix64 * _HASH_PRIMES[0])
            ^ (iy64 * _HASH_PRIMES[1])
            ^ (iz64 * _HASH_PRIMES[2])
        )
        hashed = (hashed & 0x7FFFFFFF) % self.table_sizes  # broadcast over L

        # Dense indices where dense_mask is True.
        N_l = self.res_i  # (L,)
        dense_idx = (ix64 * N_l + iy64) * N_l + iz64
        local_idx = torch.where(self.dense_mask, dense_idx, hashed)
        global_idx = local_idx + self.offsets  # broadcast

        flat = self.embeddings.index_select(0, global_idx.reshape(-1))
        return flat.reshape(*global_idx.shape, self.feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode (..., 3) -> (..., n_levels * feat_dim)."""
        orig_shape = x.shape
        x = x.reshape(-1, 3)
        N = x.shape[0]
        L = self.n_levels

        # Map [-b, b] -> [0, 1].
        u = ((x + self.bbox_half) / (2 * self.bbox_half)).clamp(0.0, 1.0)
        # (N, L, 3) coordinate per level.
        p = u.unsqueeze(1) * (self.res_f.view(1, L, 1) - 1.0)
        # Voxel-relative weights.
        p_floor = p.floor()
        w = p - p_floor  # (N, L, 3) in [0, 1]
        ix0 = p_floor[..., 0].to(torch.int64).clamp(min=0)
        iy0 = p_floor[..., 1].to(torch.int64).clamp(min=0)
        iz0 = p_floor[..., 2].to(torch.int64).clamp(min=0)
        # Clamp against the per-level resolution.
        N_l_m1 = (self.res_i - 1).view(1, L)
        ix0 = torch.minimum(ix0, N_l_m1)
        iy0 = torch.minimum(iy0, N_l_m1)
        iz0 = torch.minimum(iz0, N_l_m1)
        ix1 = torch.minimum(ix0 + 1, N_l_m1)
        iy1 = torch.minimum(iy0 + 1, N_l_m1)
        iz1 = torch.minimum(iz0 + 1, N_l_m1)

        # 8 cube corners.
        f000 = self._lookup_corner(ix0, iy0, iz0)
        f001 = self._lookup_corner(ix0, iy0, iz1)
        f010 = self._lookup_corner(ix0, iy1, iz0)
        f011 = self._lookup_corner(ix0, iy1, iz1)
        f100 = self._lookup_corner(ix1, iy0, iz0)
        f101 = self._lookup_corner(ix1, iy0, iz1)
        f110 = self._lookup_corner(ix1, iy1, iz0)
        f111 = self._lookup_corner(ix1, iy1, iz1)

        wx = w[..., 0:1]
        wy = w[..., 1:2]
        wz = w[..., 2:3]
        c00 = f000 * (1 - wz) + f001 * wz
        c01 = f010 * (1 - wz) + f011 * wz
        c10 = f100 * (1 - wz) + f101 * wz
        c11 = f110 * (1 - wz) + f111 * wz
        c0 = c00 * (1 - wy) + c01 * wy
        c1 = c10 * (1 - wy) + c11 * wy
        encoded = c0 * (1 - wx) + c1 * wx  # (N, L, F)
        encoded = encoded.reshape(N, L * self.feat_dim)
        return encoded.reshape(*orig_shape[:-1], self.out_dim)


class DDFHashGrid(nn.Module):
    """DDF with hash-grid positional encoding + small MLP.

    Default: 16 levels × 2 features = 32-dim hash code; sinusoidal direction
    encoding (3 + 24 = 27 dims at ``dir_freqs=4``); concat → 2 × 64 hidden →
    (distance, vis_logit).
    """

    def __init__(
        self,
        dir_freqs: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        n_levels: int = 16,
        feat_dim: int = 2,
        log2_table_size: int = 19,
        base_res: int = 16,
        growth: float = 1.5,
        bbox_half: float = 1.2,
    ):
        super().__init__()
        self.dir_freqs = dir_freqs
        self.bbox_half = bbox_half

        self.pos_encoder = HashGridEncoder(
            n_levels=n_levels,
            feat_dim=feat_dim,
            log2_table_size=log2_table_size,
            base_res=base_res,
            growth=growth,
            bbox_half=bbox_half,
        )

        pos_dim = self.pos_encoder.out_dim
        dir_dim = 3 + 3 * 2 * dir_freqs
        in_dim = pos_dim + dir_dim

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(*layers)
        self.dist_head = nn.Linear(hidden_dim, 1)
        self.vis_head = nn.Linear(hidden_dim, 1)

    def forward(self, points: torch.Tensor, dirs: torch.Tensor):
        p = self.pos_encoder(points)
        d = sinusoidal_encoding(dirs, self.dir_freqs)
        h = self.trunk(torch.cat([p, d], dim=-1))
        dist = torch.nn.functional.softplus(self.dist_head(h)).squeeze(-1)
        vis_logit = self.vis_head(h).squeeze(-1)
        return dist, vis_logit
