"""Re-orient a gaussians.pt so the scene's up-axis becomes +y (the GSSupervisor's
world_up convention). Rotates means AND quaternions (Gaussian orientations).

Default: rotate -90 deg about x, mapping (x,y,z) -> (x, z, -y), i.e. a z-up scene
becomes y-up. Needed because the garden GS frame is z-up (height in z) while the
supervisor samples y-up cameras -> otherwise the flat table is seen edge-on and
the DDF geometry comes out fragmented.
"""
import argparse, numpy as np, torch


def quat_mul(q1, q2):
    """Hamilton product of (w,x,y,z) quaternions; q1 (4,), q2 (N,4) -> (N,4)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    g = torch.load(args.inp, map_location="cpu", weights_only=False)

    # R_x(-90): (x,y,z)->(x,z,-y); old +z (up) -> new +y (up)
    R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
    qR = np.array([np.cos(-np.pi/4), np.sin(-np.pi/4), 0, 0], dtype=np.float32)  # (w,x,y,z)

    means = g["means"].numpy().astype(np.float32)
    g["means"] = torch.from_numpy(means @ R.T)
    quats = g["quats"].numpy().astype(np.float32)
    quats = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-8)
    g["quats"] = torch.from_numpy(quat_mul(qR, quats))
    m = g["means"].numpy()
    g["bbox_min"] = torch.from_numpy(m.min(0).astype(np.float32))
    g["bbox_max"] = torch.from_numpy(m.max(0).astype(np.float32))
    torch.save(g, args.out)
    print(f"re-oriented -> {args.out}  bbox_min={g['bbox_min'].tolist()} bbox_max={g['bbox_max'].tolist()}")


if __name__ == "__main__":
    main()
