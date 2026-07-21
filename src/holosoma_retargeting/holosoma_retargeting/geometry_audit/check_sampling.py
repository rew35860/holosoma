#!/usr/bin/env python
"""Object surface-point sampling coverage check for the interaction mesh.

The interaction-mesh retargeter puts a fixed set of OBJECT surface points into the
mesh. If those points are CLUSTERED (area-weighted random sampling can bunch up on
large faces) the mesh only "sees" part of the object and contact is poorly captured.
OmniRetarget avoids this with trimesh.sample.sample_surface_even (Poisson-disk-style,
= spread out); this script compares that and farthest-point sampling (FPS) against the
plain area-weighted sampling we currently use, and reports COVERAGE so we can pick.

We do NOT modify the solver here -- this is a read-only diagnostic. Run it on a few
objects/point-counts, look at the numbers + the PNG, then we choose the sampler.

Usage:
    python wuji/check_sampling.py <object.obj> [--k 100] [--out renders/sampling.png]
"""
import sys
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def farthest_point_sampling(pts, k, seed=0):
    """Pick k points that are as SPREAD OUT as possible (greedy farthest-point).

    Start from one random point; repeatedly add the candidate that is FARTHEST from
    everything picked so far. Guarantees large minimum spacing -> no clusters, even
    coverage. `pts` is a dense candidate cloud (N,3); returns the chosen indices (k,)."""
    n = len(pts)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n))
    idx = [first]
    dist = np.linalg.norm(pts - pts[first], axis=1)      # dist from every pt to the picked set
    for _ in range(k - 1):
        i = int(dist.argmax())                           # the currently-least-covered point
        idx.append(i)
        dist = np.minimum(dist, np.linalg.norm(pts - pts[i], axis=1))   # update nearest-picked dist
    return np.array(idx)


def coverage_metrics(samples, reference):
    """How well does `samples` cover the object? Compared against a DENSE `reference`
    cloud that stands in for the true surface.

    Returns (gap_max, gap_mean, min_spacing), all in the mesh's units (metres):
      gap_max  = worst uncovered spot = max over reference points of distance to the
                 nearest sample. SMALLER = better coverage (no holes).
      gap_mean = average such distance.
      min_spacing = smallest distance between two samples. LARGER = better spread
                 (no clusters). Area-weighted sampling makes this tiny.
    """
    d_ref, _ = cKDTree(samples).query(reference)         # nearest sample for each surface point
    d_pair, _ = cKDTree(samples).query(samples, k=2)     # nearest OTHER sample for each sample
    return d_ref.max(), d_ref.mean(), d_pair[:, 1].min()


def main():
    obj = sys.argv[1]
    k = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 100
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "renders/sampling.png"

    mesh = trimesh.load(obj, force="mesh")
    print(f"object: {obj}")
    print(f"  watertight={mesh.is_watertight}  verts={len(mesh.vertices)}  faces={len(mesh.faces)}")
    print(f"  size(cm): {mesh.bounding_box.extents * 100}")

    # dense reference cloud = stand-in for the true continuous surface
    np.random.seed(42)                                           # reproducible sampling
    dense, _ = trimesh.sample.sample_surface_even(mesh, 8000, seed=42)
    dense = np.asarray(dense)

    # three ways to pick k object points (even uses seed=42 -> IDENTICAL to load_object_data / the solve)
    np.random.seed(42)
    area, _ = trimesh.sample.sample_surface(mesh, k)             # current: area-weighted (can cluster)
    even, _ = trimesh.sample.sample_surface_even(mesh, k, seed=42)  # OmniRetarget/load_object_data (58 on table)
    fps = dense[farthest_point_sampling(dense, k)]               # farthest-point (max spread)
    methods = [("area-weighted (current)", np.asarray(area)),
               ("even / OmniRetarget", np.asarray(even)),
               ("farthest-point (FPS)", fps)]

    print(f"\nsampling {k} points -- coverage (cm):")
    print(f"  {'method':26s} {'gap_max':>8s} {'gap_mean':>9s} {'min_spacing':>12s}  (n_actual)")
    for name, s in methods:
        gmax, gmean, mspace = coverage_metrics(s, dense)
        print(f"  {name:26s} {gmax*100:8.2f} {gmean*100:9.2f} {mspace*100:12.2f}  ({len(s)})")
    print("  -> lower gap_max = fewer holes; higher min_spacing = less clustering")

    # visual: render the ACTUAL object surface (gray triangles, true proportions) with the
    # chosen points (red) on top -> you can recognize the object and see clustering/gaps.
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    V = np.asarray(mesh.vertices)
    Fc = np.asarray(mesh.faces)
    ext = np.array(mesh.bounding_box.extents, dtype=float)       # real x,y,z size -> undistorted aspect (writable)
    fig = plt.figure(figsize=(16, 6.5))
    for j, (name, s) in enumerate(methods):
        ax = fig.add_subplot(1, 3, j + 1, projection="3d")
        ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=Fc,   # the real mesh surface
                        color="lightgray", alpha=0.30, edgecolor="none", linewidth=0)
        ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=22, c="crimson", depthshade=False)
        ax.set_box_aspect(ext)                                    # keep the object's true shape
        ax.axis("off")
        fig.text((j + 0.5) / 3.0, 0.06, f"{name}  ({len(s)} pts)",  # caption BELOW each panel
                 ha="center", fontsize=13, weight="bold")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.11, wspace=0.0)
    plt.savefig(out, dpi=110); print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
