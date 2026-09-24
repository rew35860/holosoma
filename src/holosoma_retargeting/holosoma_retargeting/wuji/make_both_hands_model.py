#!/usr/bin/env python
"""Build a TWO-HAND floating model: left + right Wuji hands, each on its own named 6-DOF root, + the table
(CoACD pieces). Same recipe as make_hand_model.py but with both hands, so TopoRetargeter can be run once per
side (it discovers {side}_wrist_* + {prefix}finger* by name). Object added LAST -> its 7 qpos sit at the tail.

qpos layout: left root(6) + left fingers(20) + right root(6) + right fingers(20) + object(7) = 59.

Usage (omniretarget env):
    WUJI_HAND_DESCRIPTION=~/Downloads/wuji-hand-description python wuji/make_both_hands_model.py table --coacd
        -> models/g1/hand_both_w_table_coacd.xml
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mujoco

REPO = Path(__file__).resolve().parent.parent
G1DIR = REPO / "models/g1"
ASSETS = str(G1DIR / "assets") + "/"
WUJI = Path(os.environ.get("WUJI_HAND_DESCRIPTION", Path.home() / "Downloads/wuji-hand-description"))
PREFIX = {"left": "wj_", "right": "wjr_"}


def make_both(obj_name: str = "table", coacd: bool = True, ball: bool = True, collide: bool = True):
    """Build the two-hand floating model.

    Args:
        obj_name (str): interaction object (needs models/<obj>/<obj>.obj and, for coacd, models/decomp/<obj>/).
        coacd (bool): add the CoACD collision pieces (auto-generalizable per object).
        ball (bool): quaternion root -- 3 slides + 1 BALL joint (S3, no gimbal, TopoRetargeter base_manifold=True);
            else the legacy 3-slide + 3-hinge Euler root.
        collide (bool): turn the object CoACD collision ON (contype/conaffinity=1) so the fingers physically
            collide with the object. The Wuji hand's own collision geoms come enabled from its mjcf.
    """
    spec = mujoco.MjSpec()
    spec.meshdir = ASSETS
    spec.compiler.degree = False                              # radians (the degree default bit once)
    getattr(spec.visual, "global_").offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
    axes = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}

    for side, xpos in (("left", [-0.2, 0.0, 0.8]), ("right", [0.2, 0.0, 0.8])):
        root = spec.worldbody.add_body(name=f"hand_root_{side}", pos=xpos)
        for a in ("x", "y", "z"):                             # 3 translation slides (both root types)
            j = root.add_joint(name=f"{side}_wrist_{a}_joint", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axes[a])
            j.range = [-3.0, 3.0]
        if ball:                                              # quaternion rotation (no gimbal)
            root.add_joint(name=f"{side}_wrist_ball_joint", type=mujoco.mjtJoint.mjJNT_BALL)
        else:                                                 # legacy Euler rotation
            for a, ax in (("roll", "x"), ("pitch", "y"), ("yaw", "z")):
                j = root.add_joint(name=f"{side}_wrist_{a}_joint", type=mujoco.mjtJoint.mjJNT_HINGE, axis=axes[ax])
                j.range = [-6.28, 6.28]
        hand = mujoco.MjSpec.from_file(str(WUJI / f"mjcf/{side}.xml"))
        hand.meshdir = ASSETS
        fr = root.add_frame()
        spec.attach(hand, prefix=PREFIX[side], frame=fr)

    mesh = REPO / f"models/{obj_name}/{obj_name}.obj"
    spec.add_mesh(name=f"{obj_name}_mesh", file=str(mesh))
    b = spec.worldbody.add_body(name=f"{obj_name}_link", pos=[0.5, 0.0, 0.3])
    b.add_freejoint()
    gm = b.add_geom(); gm.name = f"{obj_name}_visual"         # visible mesh (never collides)
    gm.type = mujoco.mjtGeom.mjGEOM_MESH; gm.meshname = f"{obj_name}_mesh"
    gm.contype = 0; gm.conaffinity = 0; gm.rgba = [0.8, 0.5, 0.3, 1.0]
    if coacd:
        ct = 1 if collide else 0                             # CoACD collision ON when collide=True
        pieces = sorted((REPO / f"models/decomp/{obj_name}").glob("piece_*.obj"))
        for i, pf in enumerate(pieces):
            spec.add_mesh(name=f"{obj_name}_piece_{i:03d}_mesh", file=str(pf))
            pg = b.add_geom(); pg.name = f"{obj_name}_piece_{i:03d}"
            pg.type = mujoco.mjtGeom.mjGEOM_MESH; pg.meshname = f"{obj_name}_piece_{i:03d}_mesh"
            pg.contype = ct; pg.conaffinity = ct; pg.rgba = [0.3, 0.6, 0.9, 0.0]

    spec.compile()
    out = G1DIR / f"hand_both_w_{obj_name}{'_coacd' if coacd else ''}{'_ball' if ball else ''}.xml"
    out.write_text(spec.to_xml())
    m = mujoco.MjModel.from_xml_path(str(out))
    for side in ("left", "right"):                            # fail loud if a root is missing / in degrees
        rj = f"{side}_wrist_ball_joint" if ball else f"{side}_wrist_roll_joint"
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, rj)
        assert j >= 0 and (ball or m.jnt_range[j][1] > 6.0), f"{side} root missing/degrees"
    ncol = sum(1 for i in range(m.ngeom) if f"{obj_name}_piece" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
               and (m.geom_contype[i] or m.geom_conaffinity[i]))
    print(f"[make_both] wrote {out.name}: nq={m.nq} | root={'ball(quat)' if ball else 'euler'} | "
          f"object CoACD collidable={ncol}")
    return out


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    make_both(pos[0] if pos else "table", coacd="--coacd" in sys.argv,
              ball="--euler" not in sys.argv, collide="--no-collide" not in sys.argv)
