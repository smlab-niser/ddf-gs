"""RT-core BVH ray-query benchmark using Mitsuba 3 (OptiX backend).

Variant cuda_ad_rgb uses OptiX, which on RTX hardware (Turing+) uses dedicated
RT cores for BVH traversal. This is the actual RT-core measurement we want for
the ddf-gs paper's Sec 4.3 scaling claim.

Per-Gaussian proxy: octahedron (6 verts, 8 tris), half-extent R=0.05.
Same per-Gaussian byte footprint as the embree comparison (336 B/Gaussian),
same sweep dimensions, JSON schema compatible with bench_ddf_vs_bvh.json.

Usage:  python scripts/bench_ddf_vs_bvh_mitsuba.py
Output: /tmp/bench_rt.json
"""
from __future__ import annotations

import json
import os
import time
import numpy as np

import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import drjit as dr  # noqa: E402

print(f"Mitsuba {mi.__version__} | variant={mi.variant()} | drjit={dr.__version__}", flush=True)

N_GAUSSIANS = [5_000, 50_000, 200_000, 1_000_000]
N_RAYS = [1_024, 10_240, 102_400, 1_048_576]
REPS = 10
WARMUPS = 2
R = 0.05

OCT_V = np.array([
    [+R, 0, 0], [-R, 0, 0],
    [0, +R, 0], [0, -R, 0],
    [0, 0, +R], [0, 0, -R],
], dtype=np.float32)
OCT_F = np.array([
    [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
    [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
], dtype=np.int32)


def build_mesh_arrays(N_g, rng):
    centers = rng.uniform(-1.0, 1.0, size=(N_g, 3)).astype(np.float32)
    verts = (centers[:, None, :] + OCT_V[None, :, :]).reshape(-1, 3).astype(np.float32)
    faces = (OCT_F[None, :, :] + (np.arange(N_g, dtype=np.int32)[:, None, None] * 6)).reshape(-1, 3).astype(np.int32)
    return verts, faces


def write_ply_binary(path, verts, faces):
    N_v, N_f = verts.shape[0], faces.shape[0]
    hdr = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {N_v}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {N_f}\nproperty list uchar int vertex_indices\n"
        f"end_header\n"
    )
    face_blob = np.empty(N_f, dtype=[("cnt", "u1"), ("v", "3i4")])
    face_blob["cnt"] = 3
    face_blob["v"] = faces
    with open(path, "wb") as f:
        f.write(hdr.encode("ascii"))
        f.write(verts.astype(np.float32).tobytes())
        f.write(face_blob.tobytes())


def time_query(scene, o_np, d_np, reps):
    N = o_np.shape[0]
    o = mi.Point3f(o_np[:, 0], o_np[:, 1], o_np[:, 2])
    d = mi.Vector3f(d_np[:, 0], d_np[:, 1], d_np[:, 2])
    rays = mi.Ray3f(o, d)
    for _ in range(WARMUPS):
        si = scene.ray_intersect_preliminary(rays)
        dr.eval(si.t)
        dr.sync_thread()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        si = scene.ray_intersect_preliminary(rays)
        dr.eval(si.t)
        dr.sync_thread()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    rng = np.random.default_rng(0)
    out = {
        "n_gaussians_sweep": N_GAUSSIANS,
        "n_rays_sweep": N_RAYS,
        "bvh_rt": {
            "us_per_ray": {},
            "bytes_per_gaussian": 336,
            "platform": f"RTX 2080 Ti (1st-gen RT cores), Mitsuba {mi.__version__} cuda_ad_rgb (OptiX)",
            "octahedron_R": R,
            "reps": REPS,
            "warmups": WARMUPS,
        },
    }

    for Ng in N_GAUSSIANS:
        print(f"\n=== N_g = {Ng:,} ===", flush=True)
        verts, faces = build_mesh_arrays(Ng, rng)
        ply_path = f"/tmp/oct_{Ng}.ply"
        write_ply_binary(ply_path, verts, faces)
        print(f"  wrote {ply_path} ({os.path.getsize(ply_path) / 1e6:.1f} MB)", flush=True)

        scene = mi.load_dict({
            "type": "scene",
            "mesh": {"type": "ply", "filename": ply_path, "face_normals": True},
        })
        print(f"  built scene: {verts.shape[0]:,} verts, {faces.shape[0]:,} tris", flush=True)

        per_ng = {}
        for Nr in N_RAYS:
            o_np = rng.standard_normal((Nr, 3)).astype(np.float32)
            o_np = o_np / np.linalg.norm(o_np, axis=1, keepdims=True) * 2.0
            tgt = rng.uniform(-1, 1, size=(Nr, 3)).astype(np.float32)
            d_np = tgt - o_np
            d_np = d_np / np.linalg.norm(d_np, axis=1, keepdims=True)

            try:
                t_med = time_query(scene, o_np, d_np, REPS)
                us = t_med * 1e6 / Nr
                per_ng[str(Nr)] = us
                print(f"  N_rays={Nr:>9,}: median {t_med*1e3:9.3f} ms  =>  {us:8.4f} us/ray", flush=True)
            except Exception as e:
                print(f"  N_rays={Nr:>9,}: FAILED {e!r}", flush=True)
                per_ng[str(Nr)] = None

        out["bvh_rt"]["us_per_ray"][str(Ng)] = per_ng
        try:
            os.remove(ply_path)
        except OSError:
            pass

    with open("/tmp/bench_rt.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote /tmp/bench_rt.json")


if __name__ == "__main__":
    main()
