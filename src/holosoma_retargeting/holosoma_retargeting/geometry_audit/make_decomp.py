"""
Convex decomposition of object meshes (CoACD) for an accurate MuJoCo collision proxy.

MuJoCo collides a `type="mesh"` geom as its single convex hull, which balloons past concave objects
(see wuji/hull_audit.py). CoACD splits a mesh into many convex pieces whose union hugs the true
surface; each piece is itself convex, so MuJoCo's convex collision path is reused unchanged. This
module runs CoACD on one or more objects, measures how far the decomposed proxy still bulges past
the true surface (before: 1 hull, after: N pieces), renders a [true | proxy | overlay] row per
object, and writes the pieces to models/decomp/<label>/piece_XXX.obj for the model-rebuild step.

Usage:
    python wuji/make_decomp.py out.png [--threshold 0.05] label1=obj1.obj label2=obj2.obj ...
"""

from __future__ import annotations

import os
import sys

import coacd  # type: ignore[import-not-found]
import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def convex_planes(piece):
    """
    Compute the outward face planes of a convex piece.

    Args:
        piece (trimesh.Trimesh): A convex mesh piece.

    Returns:
        tuple: (normals, offsets) - a point p is inside the piece iff max(normals @ p - offsets) < 0.
    """
    normals = piece.face_normals
    offsets = (normals * piece.vertices[piece.faces[:, 0]]).sum(1)
    return normals, offsets


def decompose_mesh(mesh, threshold):
    """
    Convex-decompose a mesh into convex pieces with CoACD.

    Args:
        mesh (trimesh.Trimesh): The object mesh to decompose.
        threshold (float): CoACD concavity threshold; lower keeps splitting for a tighter fit.

    Returns:
        list: Convex trimesh.Trimesh pieces whose union approximates the mesh.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    parts = coacd.run_coacd(coacd.Mesh(vertices, faces), threshold=threshold)
    return [trimesh.Trimesh(np.asarray(v), np.asarray(f), process=False) for v, f in parts]


def outer_surface_bulge(pieces, true_mesh, n_per=1500, eps=1e-3):
    """
    Measure how far the decomposed proxy's outer surface bulges past the true surface.

    Samples each piece's surface, drops points buried inside another (convex) piece so only the
    union's outer boundary remains, then measures the distance from those points to the true
    surface. Matches the metric in wuji/hull_audit.py so hull-vs-decomposition is comparable.

    Args:
        pieces (list): Convex trimesh.Trimesh pieces from CoACD.
        true_mesh (trimesh.Trimesh): The original (concave) object mesh.
        n_per (int): Surface samples per piece.
        eps (float): Inside-margin (m) used to discard points buried in another piece.

    Returns:
        np.ndarray: Per-point distance (m) from the proxy's outer surface to the true surface.
    """
    planes = [convex_planes(p) for p in pieces]
    outer = []
    for i, piece in enumerate(pieces):
        samples = np.asarray(trimesh.sample.sample_surface(piece, n_per)[0])
        buried = np.zeros(len(samples), bool)
        for j, (normals, offsets) in enumerate(planes):
            if j != i:
                buried |= (samples @ normals.T - offsets).max(1) < -eps
        outer.append(samples[~buried])
    _, dist, _ = trimesh.proximity.closest_point(true_mesh, np.vstack(outer))
    return dist


def audit_object(label, path, threshold):
    """
    Decompose one object and gather its before/after collision-proxy statistics.

    Runs CoACD, saves the pieces to models/decomp/<label>/, and measures the surface bulge for both
    the single convex hull (before) and the decomposed pieces (after).

    Args:
        label (str): Short object name; also the output sub-directory for the pieces.
        path (str): Path to the object mesh (.obj).
        threshold (float): CoACD concavity threshold.

    Returns:
        dict: Mesh, pieces, bounding-box extents, and hull/decomp bulge max+mean (cm).
    """
    mesh = trimesh.load(path, force="mesh")
    pieces = decompose_mesh(mesh, threshold)

    hull_samples = np.asarray(trimesh.sample.sample_surface(mesh.convex_hull, 4000)[0])
    hull_dist = trimesh.proximity.closest_point(mesh, hull_samples)[1]  # single-hull bulge (before)
    decomp_dist = outer_surface_bulge(pieces, mesh)                     # decomposed bulge (after)

    out_dir = os.path.join("models", "decomp", label)
    os.makedirs(out_dir, exist_ok=True)
    for i, piece in enumerate(pieces):
        piece.export(os.path.join(out_dir, f"piece_{i:03d}.obj"))

    row = dict(label=label, mesh=mesh, pieces=pieces, ext=np.array(mesh.bounding_box.extents, dtype=float),
               hull_max=hull_dist.max() * 100, hull_mean=hull_dist.mean() * 100,
               dec_max=decomp_dist.max() * 100, dec_mean=decomp_dist.mean() * 100)
    print(f"{label:22s} {len(pieces):2d} pieces | hull bulge max {row['hull_max']:5.1f}cm mean {row['hull_mean']:4.1f}cm"
          f"  ->  decomp bulge max {row['dec_max']:5.1f}cm mean {row['dec_mean']:4.1f}cm   (saved -> {out_dir}/)")
    return row


def render_rows(rows, out):
    """
    Render one [true mesh | decomposed pieces | overlay] row per object to a single figure.

    Args:
        rows (list): The dicts returned by audit_object.
        out (str): Output image path.
    """
    n = len(rows)
    fig = plt.figure(figsize=(13, 4.0 * n))
    titles = ["visual mesh (true shape)", "collision proxy (CoACD pieces)", "overlay"]
    cmap = plt.get_cmap("tab20")
    for r, row in enumerate(rows):
        verts, faces = np.asarray(row["mesh"].vertices), np.asarray(row["mesh"].faces)
        for c in range(3):
            ax = fig.add_subplot(n, 3, r * 3 + c + 1, projection="3d")
            if c in (0, 2):
                ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                                color="dimgray", alpha=1.0 if c == 2 else 0.95, edgecolor="none")
            if c in (1, 2):
                for k, piece in enumerate(row["pieces"]):
                    pv, pf = np.asarray(piece.vertices), np.asarray(piece.faces)
                    ax.plot_trisurf(pv[:, 0], pv[:, 1], pv[:, 2], triangles=pf,
                                    color=cmap(k % 20), alpha=0.35 if c == 2 else 0.85, edgecolor="none")
            ax.set_box_aspect(row["ext"])
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=12, weight="bold", pad=0)
        fig.text(0.015, 1 - (r + 0.5) / n,
                 f"{row['label']}\n{len(row['pieces'])} pieces\nbulge max\n{row['hull_max']:.0f}->{row['dec_max']:.0f}cm\n"
                 f"mean {row['hull_mean']:.0f}->{row['dec_mean']:.0f}cm",
                 va="center", ha="left", fontsize=10.5, weight="bold")

    fig.subplots_adjust(left=0.11, right=1.0, top=0.97, bottom=0.0, wspace=0.0, hspace=0.0)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    plt.savefig(out, dpi=115)
    print(f"wrote {out}")


def main():
    coacd.set_log_level("error")
    out = sys.argv[1]
    threshold = float(sys.argv[sys.argv.index("--threshold") + 1]) if "--threshold" in sys.argv else 0.05
    items = [a.split("=", 1) for a in sys.argv[2:] if "=" in a and not a.startswith("--")]
    rows = [audit_object(label, path, threshold) for label, path in items]
    render_rows(rows, out)


if __name__ == "__main__":
    main()
