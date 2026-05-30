"""Aggregate Stage-3 metrics across all objects into one table."""

import json
from pathlib import Path


def main():
    rows = []
    for d in sorted(Path("runs").glob("*_ddf*/stage3/metrics.json")):
        rows.append(json.loads(d.read_text()))
    if not rows:
        print("no stage3 metrics found.")
        return
    print(f"{'obj':<10s}  {'verts':>8s}  {'faces':>8s}  {'CD mean':>10s}  {'CD med':>10s}")
    for r in sorted(rows, key=lambda x: x["obj"]):
        print(f"{r['obj']:<10s}  {r['n_verts']:>8d}  {r['n_faces']:>8d}  "
              f"{r['chamfer_mean']:>10.4f}  {r['chamfer_median']:>10.4f}")
    cds = [r["chamfer_mean"] for r in rows]
    print(f"\n{'mean':>10s}                              {sum(cds)/len(cds):>10.4f}")


if __name__ == "__main__":
    main()
