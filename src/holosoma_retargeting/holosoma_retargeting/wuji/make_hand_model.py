#!/usr/bin/env python
"""Build a FLOATING Wuji hand + object model — the hand-only experiment, no G1 body.

The full-body pipeline entangles three failure sources: Stage A's body fit, the arm's reachability,
and the hand-object solve itself. This model removes the first two by construction: the hand hangs
from a 6-DOF root (3 slide + 3 hinge) in empty space, so the retargeter can place it ANYWHERE --
wherever the human hand was -- and the only remaining question is whether the hand-object objective
can produce a grab. That is the question worth isolating.

The root joints are deliberately NAMED like the G1 wrist joints ({side}_wrist_{x,y,z,roll,pitch,yaw}
_joint) so TopoRetargeter's by-name DOF discovery picks them up with no solver changes: it frees
6 base DOF + 20 fingers = 26, instead of the G1's 3 + 20. Explicit slide/hinge joints rather than a
freejoint because the SQP integrates plain `q += dq` -- a freejoint's quaternion cannot be updated
that way. (Three stacked hinges = Euler angles; fine here, the palm orientation stays far from
gimbal lock for a tabletop grasp.)

The object is added exactly as in make_wuji_model.py: visual mesh + freejoint (LAST, so its 7 qpos
sit at the tail where every driver reads them) + CoACD piece geoms for the soft-pen constraint.

Usage (omniretarget env):
    python wuji/make_hand_model.py table [--side right] [--coacd]
        -> models/g1/hand_<side>_w_<obj>[_coacd].xml
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mujoco

REPO = Path(__file__).resolve().parent.parent
G1DIR = REPO / "models/g1"
ASSETS = str(G1DIR / "assets") + "/"
WUJI = Path(os.environ.get("WUJI_HAND_DESCRIPTION", Path.home() / "wuji-hand-description"))
PREFIX = {"left": "wj_", "right": "wjr_"}                 # match the G1 models' (inconsistent) prefixes


def make(obj_name: str, side: str = "right", coacd: bool = False):
    """Assemble the floating-hand model and write it next to the G1 ones.

    Args:
        obj_name: object under models/<obj>/<obj>.obj, e.g. "table".
        side: which hand.
        coacd: add the CoACD piece geoms (from models/decomp/<obj>/) for accurate non-penetration;
            without it the object's single mesh geom is the (inflated) convex-hull proxy.

    Returns:
        Path: the written xml.
    """
    spec = mujoco.MjSpec()
    spec.meshdir = ASSETS
    # MJCF compiles angles as DEGREES by default -- a hinge range of [-6.28, 6.28] silently becomes
    # +-6.28 deg (0.11 rad), which made the very first QP infeasible. Everything here is radians.
    spec.compiler.degree = False
    getattr(spec.visual, "global_").offwidth = 1280       # 'global' is a Python keyword
    spec.visual.global_.offheight = 960
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])

    # 6-DOF root: 3 slides + 3 hinges, named so TopoRetargeter discovers them as free base DOF.
    root = spec.worldbody.add_body(name="hand_root", pos=[0, 0, 0.8])
    axes = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
    for a in ("x", "y", "z"):
        j = root.add_joint(name=f"{side}_wrist_{a}_joint", type=mujoco.mjtJoint.mjJNT_SLIDE,
                           axis=axes[a])
        j.range = [-3.0, 3.0]                             # generous; lo/hi feed the QP box constraint
    for a, ax in (("roll", "x"), ("pitch", "y"), ("yaw", "z")):
        j = root.add_joint(name=f"{side}_wrist_{a}_joint", type=mujoco.mjtJoint.mjJNT_HINGE,
                           axis=axes[ax])
        j.range = [-6.28, 6.28]

    hand = mujoco.MjSpec.from_file(str(WUJI / f"mjcf/{side}.xml"))
    hand.meshdir = ASSETS
    fr = root.add_frame()                                 # identity: palm frame == root frame
    spec.attach(hand, prefix=PREFIX[side], frame=fr)

    # object: visual mesh + optional CoACD pieces on a freejoint body -- same recipe as
    # make_wuji_model.py, and added LAST so the object's 7 qpos are at the tail of qpos.
    mesh = REPO / f"models/{obj_name}/{obj_name}.obj"
    if not mesh.is_file():
        sys.exit(f"object mesh not found: {mesh}")
    spec.add_mesh(name=f"{obj_name}_mesh", file=str(mesh))
    b = spec.worldbody.add_body(name=f"{obj_name}_link", pos=[0.5, 0.0, 0.3])
    b.add_freejoint()
    gm = b.add_geom()
    gm.type = mujoco.mjtGeom.mjGEOM_MESH
    gm.meshname = f"{obj_name}_mesh"
    gm.contype = 0
    gm.conaffinity = 0                                    # constraint via mj_geomDistance, not contacts
    if coacd:
        pieces = sorted((REPO / f"models/decomp/{obj_name}").glob("piece_*.obj"))
        if not pieces:
            sys.exit(f"no CoACD pieces in models/decomp/{obj_name} -- run make_wuji_model --coacd once")
        for i, pf in enumerate(pieces):
            spec.add_mesh(name=f"{obj_name}_piece_{i:03d}_mesh", file=str(pf))
            pg = b.add_geom()
            pg.name = f"{obj_name}_piece_{i:03d}"
            pg.type = mujoco.mjtGeom.mjGEOM_MESH
            pg.meshname = f"{obj_name}_piece_{i:03d}_mesh"
            pg.contype = 0
            pg.conaffinity = 0
            pg.rgba = [0.3, 0.6, 0.9, 0.0]

    spec.compile()
    out = G1DIR / f"hand_{side}_w_{obj_name}{'_coacd' if coacd else ''}.xml"
    out.write_text(spec.to_xml())
    m = mujoco.MjModel.from_xml_path(str(out))
    # verify the root ranges survived compilation IN RADIANS -- the degree default already bit once
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wrist_roll_joint")
    assert m.jnt_range[j][1] > 6.0, f"root hinge range compiled to {m.jnt_range[j]} -- degrees again?"
    print(f"[make_hand_model] wrote {out.name}: nq={m.nq} (6 root + 20 fingers + 7 object)"
          f" | root hinge range {m.jnt_range[j]}")
    return out


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    side = sys.argv[sys.argv.index("--side") + 1] if "--side" in sys.argv else "right"
    make(pos[0] if pos else "table", side=side, coacd="--coacd" in sys.argv)
