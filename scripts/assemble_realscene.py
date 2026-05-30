"""Assemble the multi-scene real-scene GI figure: stack the per-scene panels
(each a GS | DDF-GI-map | GS+DDF-GI triple) into one clean grid. No baked text
labels -- the column headers are typeset in LaTeX (paper font). Output
renders/gi_realscene.png.
"""
import numpy as np
from PIL import Image

SCENES = ["renders/gi_garden.png", "renders/gi_kitchen.png", "renders/gi_counter.png"]
HDR = 30      # per-scene header strip baked by gi_garden.py (crop it off)

panels, W = [], None
for path in SCENES:
    a = np.asarray(Image.open(path).convert("RGB"))[HDR:]
    if W is None:
        W = a.shape[1]
    elif a.shape[1] != W:
        a = np.asarray(Image.fromarray(a).resize((W, int(a.shape[0] * W / a.shape[1]))))
    panels.append(a)

sheet = np.concatenate(panels, axis=0)
# trim outer white margin (keep a uniform 14px border)
nw = (sheet < 245).any(axis=2)
ys, xs = np.where(nw)
m = 14
img = Image.fromarray(sheet[max(0, ys.min() - m):ys.max() + 1 + m,
                            max(0, xs.min() - m):xs.max() + 1 + m])
img.save("renders/gi_realscene.png")
print(f"wrote renders/gi_realscene.png  ({len(panels)} scenes x 3 cols, no labels, {img.size})")
