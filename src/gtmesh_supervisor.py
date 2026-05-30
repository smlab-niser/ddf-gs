"""GT mesh ray-cast supervisor — the 23ddf recipe (Behera & Mishra 2023).

Supervises the DDF with EXACT distances from ray-casting the ground-truth mesh.
This is the cleanest possible distance supervision: no GS noise, no NeuS
distillation, no depth-render blur. Used to test whether the DDF *architecture*
can capture thin features (legs, handles) when given perfect supervision.

Sampling (23ddf-style):
  - Surface-anchored origins: sample points on the GT surface, perturb off it,
    cast rays in random directions.
  - Outside-in origins: random points on a sphere around the object, cast rays
    toward the interior.
  - Ray-march self-consistency: marched copies of hit rays.
"""

import numpy as np
import torch
import trimesh


class GTMeshSupervisor:

    def __init__(
        self,
        gt_mesh_path: str,
        mesh_center: np.ndarray,
        mesh_scale: float,
        device: str = "cuda",
        march_ratio: float = 0.5,
        surface_ratio: float = 0.5,
        gso_rotate: bool = False,
        frustum_ratio: float = 0.0,
        image_size: int = 128,
    ):
        self.device = device
        self.march_ratio = march_ratio
        self.surface_ratio = surface_ratio
        # frustum_ratio>0 enables COMBINED supervision: a fraction of each batch
        # comes from coherent camera-frustum rays (raycast against the GT mesh)
        # for clean PRIMARY-ray tracing, the rest from omnidirectional
        # surface-anchored rays for correct SECONDARY-ray (shadow/AO) visibility.
        self.frustum_ratio = frustum_ratio
        self.image_size = image_size

        m = trimesh.load(gt_mesh_path, force="mesh", process=False)
        if isinstance(m, trimesh.Scene):
            geoms = [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
            m = trimesh.util.concatenate(geoms)
        # Normalize to match the DDF's training frame: (v - center) * scale
        m.apply_translation(-mesh_center)
        m.apply_scale(mesh_scale)
        if gso_rotate:
            # GSO objects need the X-(-90) rotation to match the camera frame
            rot = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                           dtype=np.float64)
            m.apply_transform(rot)
        self.mesh = m
        self.intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(m)
        print(f"GTMeshSupervisor: {len(m.vertices)} verts, {len(m.faces)} faces, "
              f"bounds {m.bounds.tolist()}")

    def _cast(self, origins_np, dirs_np):
        """Cast rays, return (t, hit) — distance to first hit along each ray."""
        locs, ray_idx, _ = self.intersector.intersects_location(
            origins_np, dirs_np, multiple_hits=False)
        N = origins_np.shape[0]
        t = np.full(N, -1.0, dtype=np.float32)
        hit = np.zeros(N, dtype=bool)
        if len(ray_idx) > 0:
            d = np.linalg.norm(locs - origins_np[ray_idx], axis=1)
            # Keep nearest hit per ray
            for i, ri in enumerate(ray_idx):
                if not hit[ri] or d[i] < t[ri]:
                    t[ri] = d[i]
                    hit[ri] = True
        return t, hit

    def _frustum_rays(self, n_rays: int):
        """Coherent camera-frustum rays from a random spherical camera, raycast
        against the GT mesh. Returns (origins, dirs) numpy."""
        import math as _m
        from .gs_supervisor import _spherical_camera, _pixels_to_world_rays
        elev = np.random.uniform(-30.0, 60.0)
        azim = np.random.uniform(0.0, 360.0)
        radius = np.random.uniform(2.0, 3.0)
        _, K, c2w = _spherical_camera(elev, azim, radius, self.image_size,
                                      device=self.device)
        o, d = _pixels_to_world_rays(K, c2w, self.image_size, self.device)
        o = o.reshape(-1, 3); d = d.reshape(-1, 3)
        # subsample to n_rays
        idx = torch.randperm(o.shape[0], device=self.device)[:n_rays]
        return o[idx].cpu().numpy(), d[idx].cpu().numpy()

    def sample(self, batch_size: int):
        if self.frustum_ratio > 0:
            return self._sample_combined(batch_size)
        n_surf = int(batch_size * self.surface_ratio)
        n_out = batch_size - n_surf

        # Surface-anchored rays: sample surface points, perturb, random dir
        surf_pts, _ = trimesh.sample.sample_surface(self.mesh, n_surf)
        surf_pts = np.asarray(surf_pts, dtype=np.float32)
        offset = np.random.uniform(0.01, 0.05, (n_surf, 1)).astype(np.float32)
        surf_dirs = np.random.randn(n_surf, 3).astype(np.float32)
        surf_dirs /= (np.linalg.norm(surf_dirs, axis=1, keepdims=True) + 1e-8)
        surf_origins = surf_pts + offset * surf_dirs

        # Outside-in rays: random sphere points, cast toward origin + jitter
        out_dirs_unit = np.random.randn(n_out, 3).astype(np.float32)
        out_dirs_unit /= (np.linalg.norm(out_dirs_unit, axis=1, keepdims=True) + 1e-8)
        radii = np.random.uniform(1.5, 3.0, (n_out, 1)).astype(np.float32)
        out_origins = out_dirs_unit * radii
        out_dirs = -out_dirs_unit + np.random.randn(n_out, 3).astype(np.float32) * 0.2
        out_dirs /= (np.linalg.norm(out_dirs, axis=1, keepdims=True) + 1e-8)

        origins_np = np.concatenate([surf_origins, out_origins], axis=0)
        dirs_np = np.concatenate([surf_dirs, out_dirs], axis=0)

        t_np, hit_np = self._cast(origins_np, dirs_np)

        origins = torch.from_numpy(origins_np).to(self.device)
        dirs = torch.from_numpy(dirs_np).to(self.device)
        t_gt = torch.from_numpy(np.where(hit_np, t_np, 0.0)).float().to(self.device)
        hit_gt = torch.from_numpy(hit_np).to(self.device)

        # Ray-march self-consistency
        if self.march_ratio > 0 and hit_gt.any():
            n_march = int(round(batch_size * self.march_ratio))
            n_march = min(n_march, batch_size)
            hit_idx = hit_gt.nonzero(as_tuple=True)[0]
            src = hit_idx[torch.randint(0, hit_idx.numel(), (n_march,), device=self.device)]
            src_o = origins[src]
            src_d = dirs[src]
            src_t = t_gt[src]
            s = torch.empty(n_march, device=self.device).uniform_(0.05, 0.9) * src_t
            origins[-n_march:] = src_o + s.unsqueeze(-1) * src_d
            dirs[-n_march:] = src_d
            t_gt[-n_march:] = src_t - s
            hit_gt[-n_march:] = True

        return origins, dirs, t_gt, hit_gt

    def _sample_combined(self, batch_size: int):
        """Combined supervision: frustum-camera primary rays + omnidirectional
        surface-anchored secondary rays, all with clean GT-mesh raycast distances.
        Gives one DDF that traces primary cleanly AND answers shadow rays."""
        import trimesh
        n_frust = int(batch_size * self.frustum_ratio)
        n_surf = batch_size - n_frust

        # Frustum primary rays (coherent camera) — for clean primary tracing
        fo, fd = self._frustum_rays(n_frust)

        # Omnidirectional surface-anchored rays — for secondary/shadow visibility
        surf_pts, _ = trimesh.sample.sample_surface(self.mesh, n_surf)
        surf_pts = np.asarray(surf_pts, dtype=np.float32)
        offset = np.random.uniform(0.01, 0.05, (n_surf, 1)).astype(np.float32)
        sd = np.random.randn(n_surf, 3).astype(np.float32)
        sd /= (np.linalg.norm(sd, axis=1, keepdims=True) + 1e-8)
        so = surf_pts + offset * sd

        origins_np = np.concatenate([fo, so], axis=0)
        dirs_np = np.concatenate([fd, sd], axis=0)
        t_np, hit_np = self._cast(origins_np, dirs_np)

        origins = torch.from_numpy(origins_np).to(self.device)
        dirs = torch.from_numpy(dirs_np).to(self.device)
        t_gt = torch.from_numpy(np.where(hit_np, t_np, 0.0)).float().to(self.device)
        hit_gt = torch.from_numpy(hit_np).to(self.device)

        if self.march_ratio > 0 and hit_gt.any():
            n_march = min(int(round(batch_size * self.march_ratio)), batch_size)
            hit_idx = hit_gt.nonzero(as_tuple=True)[0]
            src = hit_idx[torch.randint(0, hit_idx.numel(), (n_march,), device=self.device)]
            src_o, src_d, src_t = origins[src], dirs[src], t_gt[src]
            s = torch.empty(n_march, device=self.device).uniform_(0.05, 0.9) * src_t
            origins[-n_march:] = src_o + s.unsqueeze(-1) * src_d
            dirs[-n_march:] = src_d
            t_gt[-n_march:] = src_t - s
            hit_gt[-n_march:] = True

        return origins, dirs, t_gt, hit_gt
