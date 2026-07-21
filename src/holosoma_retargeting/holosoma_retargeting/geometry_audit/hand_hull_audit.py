#!/usr/bin/env python
"""Audit the Wuji hand collision geometry: TRUE finger mesh vs the CONVEX HULL MuJoCo collides.

The object version of this (hull_audit.py / make_decomp.py) showed a concave table ballooning into a
63 cm box, motivating CoACD. This is the same audit for the HAND, and the point is the opposite: the
finger links are near-convex capsule segments, so the convex hull MuJoCo collides them as is nearly the
true mesh (worst ~0.9 cm at the palm) -- no decomposition needed. The figure makes that visible: the
hull column looks like the true column.

Renders one [true | hull | overlay] row per selected part, same layout/style as make_decomp.render_rows
so it drops next to the object figures for a side-by-side.

Usage (omniretarget env):
    python wuji/hand_hull_audit.py            # 4 representative links -> renders/hand_hull/HAND_hull.png
    python wuji/hand_hull_audit.py --whole    # WHOLE assembled hand (all 25 links) -> HAND_whole.png
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402

MESH_DIR = os.path.expanduser("~/Downloads/wuji-hand-description/meshes/right")
HAND_XML = os.path.expanduser("~/Downloads/wuji-hand-description/mjcf/right.xml")
# a representative sample: palm (worst), a knuckle base, a contacting phalanx, a fingertip
PARTS = ["right_palm_link", "right_finger3_link1", "right_finger3_link4", "right_finger3_tip_link"]
OUT = "renders/hand_hull/HAND_hull.png"


def audit_part(name: str) -> dict:
    """Load a link mesh, build its convex hull, measure how far the hull bulges past the true surface.

    Args:
        name: STL basename (no extension) under MESH_DIR.

    Returns:
        dict: {label, mesh, hull, ext (bbox for aspect), bulge_max, bulge_mean} in cm.
    """
    m = trimesh.load(os.path.join(MESH_DIR, f"{name}.STL"), force="mesh")
    hull = m.convex_hull
    hp, _ = trimesh.sample.sample_surface(hull, 3000)                # hull surface points
    d = trimesh.proximity.ProximityQuery(m).on_surface(hp)[1]        # distance to the TRUE surface
    return dict(label=name.replace("right_", ""), mesh=m, hull=hull,
                ext=np.ptp(m.vertices, axis=0), bulge_max=d.max() * 100, bulge_mean=d.mean() * 100)


def render_rows(rows: list, out: str):
    """One [true mesh | convex hull | overlay] row per part -- mirrors make_decomp.render_rows."""
    n = len(rows)
    fig = plt.figure(figsize=(13, 4.0 * n))
    titles = ["visual mesh (true shape)", "collision proxy (convex hull = what MuJoCo collides)",
              "overlay"]
    for r, row in enumerate(rows):
        v, f = np.asarray(row["mesh"].vertices), np.asarray(row["mesh"].faces)
        hv, hf = np.asarray(row["hull"].vertices), np.asarray(row["hull"].faces)
        for c in range(3):
            ax = fig.add_subplot(n, 3, r * 3 + c + 1, projection="3d")
            if c in (0, 2):
                ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f,
                                color="dimgray", alpha=1.0 if c == 2 else 0.95, edgecolor="none")
            if c in (1, 2):
                ax.plot_trisurf(hv[:, 0], hv[:, 1], hv[:, 2], triangles=hf,
                                color="tab:orange", alpha=0.35 if c == 2 else 0.85, edgecolor="none")
            ax.set_box_aspect(row["ext"])
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=12, weight="bold", pad=0)
        fig.text(0.015, 1 - (r + 0.5) / n,
                 f"{row['label']}\nhull bulge\nmax {row['bulge_max']:.2f} cm\nmean {row['bulge_mean']:.2f} cm",
                 va="center", ha="left", fontsize=10.5, weight="bold")
    fig.subplots_adjust(left=0.11, right=1.0, top=0.97, bottom=0.0, wspace=0.0, hspace=0.0)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    plt.savefig(out, dpi=115)
    print(f"wrote {out}")


def assemble_whole_hand():
    """Pose every link at the hand's rest configuration and return the assembled true mesh + per-link
    hulls, exactly as MuJoCo sees them (each geom hulled independently, then placed by its rest pose).

    Uses MuJoCo FK at qpos=0 to get each collision geom's world transform, so the hulls sit where the
    collision engine actually puts them.

    Returns:
        tuple: (true trimesh, hull trimesh) of the whole hand.
    """
    import mujoco

    m = mujoco.MjModel.from_xml_path(HAND_XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    true_parts, hull_parts = [], []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or int(m.geom_contype[g]) == 0:
            continue                                                    # collision geoms only (skip visual)
        # use the COMPILED mesh verts (mesh_pos/mesh_quat already baked in), not the raw STL -- else
        # each link is off by MuJoCo's per-mesh recentring (measured up to 3.9 cm) and the parts detach.
        mid = int(m.geom_dataid[g])
        va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        V = m.mesh_vert[va:va + vn].astype(np.float64)
        F = m.mesh_face[fa:fa + fn].astype(np.int64)
        F = F - va if F.max() >= vn else F                              # face idx local-vs-global guard
        mesh = trimesh.Trimesh(V, F, process=False)
        T = np.eye(4)
        T[:3, :3] = d.geom_xmat[g].reshape(3, 3)                        # geom world pose places it exactly
        T[:3, 3] = d.geom_xpos[g]
        true_parts.append(mesh.copy().apply_transform(T))
        hull_parts.append(mesh.convex_hull.apply_transform(T))          # hull per link, then place it
    return trimesh.util.concatenate(true_parts), trimesh.util.concatenate(hull_parts)


def render_whole(out: str, view_azim: float = 0.0):
    """One row: whole-hand [true mesh | assembled per-link hulls | overlay].

    view_azim rotates the camera about the vertical: 0 = one face toward the screen, 180 = the other
    (palm vs back of hand).
    """
    true, hull = assemble_whole_hand()
    v, f = np.asarray(true.vertices), np.asarray(true.faces)
    hv, hf = np.asarray(hull.vertices), np.asarray(hull.faces)
    ext = np.ptp(v, axis=0)
    fig = plt.figure(figsize=(13, 5.5))
    titles = ["visual mesh (true shape)", "collision proxy (per-link convex hulls)", "overlay"]
    for c in range(3):
        ax = fig.add_subplot(1, 3, c + 1, projection="3d")
        if c in (0, 2):
            ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f,
                            color="dimgray", alpha=1.0 if c == 2 else 0.95, edgecolor="none")
        if c in (1, 2):
            ax.plot_trisurf(hv[:, 0], hv[:, 1], hv[:, 2], triangles=hf,
                            color="tab:orange", alpha=0.4 if c == 2 else 0.85, edgecolor="none")
        ax.set_box_aspect(ext)
        ax.axis("off")
        ax.view_init(elev=0, azim=view_azim)  # look along the palm normal -> whole hand faces the screen
        ax.set_title(titles[c], fontsize=12, weight="bold")
    fig.text(0.5, 0.03, "Wuji hand collision geometry: each link hulled independently -- the hulls hug "
             "the true shape (sub-cm), no decomposition needed", ha="center", fontsize=11, weight="bold")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.06, wspace=0.0)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    plt.savefig(out, dpi=130)
    print(f"wrote {out}")


def main():
    if "--whole" in sys.argv:
        render_whole("renders/hand_hull/HAND_whole.png")
        return
    if "--back" in sys.argv:
        render_whole("renders/hand_hull/HAND_whole_back.png", view_azim=180.0)
        return
    rows = [audit_part(p) for p in PARTS]
    for row in rows:
        print(f"  {row['label']:22s} hull bulge  max {row['bulge_max']:.2f}  mean {row['bulge_mean']:.2f} cm")
    render_rows(rows, OUT)


if __name__ == "__main__":
    main()
