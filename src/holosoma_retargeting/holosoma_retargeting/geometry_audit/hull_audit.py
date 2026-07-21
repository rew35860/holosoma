#!/usr/bin/env python
"""Collision-proxy audit across objects: how badly MuJoCo's CONVEX HULL over-approximates each
object's true (concave) surface — the motivation for decomposing into primitive/convex geoms.

MuJoCo collides a `type="mesh"` geom as its convex hull, so the seat-to-legs gap of a chair, the
underside of a table, or the thin pole of a lamp all get "filled in". This renders one figure with
a ROW per object (visual mesh | convex hull | overlay) and prints how far the hull bulges past the
true surface, so the whole set is comparable on a single slide.

Usage:
    python wuji/hull_audit.py out.png  label1=obj1.obj  label2=obj2.obj  ...
"""
import os
import sys

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
items = [a.split("=", 1) for a in sys.argv[2:]]                     # [(label, path), ...]

rows = []
for label, path in items:
    m = trimesh.load(path, force="mesh")
    hull = m.convex_hull
    hp, _ = trimesh.sample.sample_surface(hull, 4000)              # sample the hull surface ...
    _, d, _ = trimesh.proximity.closest_point(m, np.asarray(hp))   # ... distance to the TRUE surface
    rows.append(dict(label=label, m=m, hull=hull, dmax=d.max() * 100, dmean=d.mean() * 100,
                     ext=np.array(m.bounding_box.extents, dtype=float)))
    print(f"{label:22s} bbox(cm) {np.round(rows[-1]['ext'] * 100, 1)}  hull inflation: "
          f"max {rows[-1]['dmax']:.1f}cm  mean {rows[-1]['dmean']:.1f}cm")

n = len(rows)
fig = plt.figure(figsize=(13, 4.0 * n))
col_titles = ["visual mesh (true shape)", "collision mesh (convex hull)", "overlay"]
for r, row in enumerate(rows):
    V, F = np.asarray(row["m"].vertices), np.asarray(row["m"].faces)
    HV, HF = np.asarray(row["hull"].vertices), np.asarray(row["hull"].faces)
    for c in range(3):
        ax = fig.add_subplot(n, 3, r * 3 + c + 1, projection="3d")
        if c in (0, 2):
            ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F,
                            color="dimgray", alpha=1.0 if c == 2 else 0.95, edgecolor="none")
        if c in (1, 2):
            ax.plot_trisurf(HV[:, 0], HV[:, 1], HV[:, 2], triangles=HF,
                            color="steelblue", alpha=0.22 if c == 2 else 0.6, edgecolor="none")
        ax.set_box_aspect(row["ext"]); ax.axis("off")
        if r == 0:
            ax.set_title(col_titles[c], fontsize=12, weight="bold", pad=0)
    # row label on the left
    fig.text(0.015, 1 - (r + 0.5) / n, f"{row['label']}\nhull bulge\nmax {row['dmax']:.0f}cm\nmean {row['dmean']:.0f}cm",
             va="center", ha="left", fontsize=11, weight="bold")

fig.subplots_adjust(left=0.11, right=1.0, top=0.97, bottom=0.0, wspace=0.0, hspace=0.0)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
plt.savefig(out, dpi=115)
print(f"wrote {out}")
