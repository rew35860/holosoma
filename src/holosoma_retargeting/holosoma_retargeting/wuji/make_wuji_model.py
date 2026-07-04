#!/usr/bin/env python
"""Generate the G1 + Wuji (both hands) MuJoCo model, optionally with an object baked in.

OmniRetarget's object-interaction loads a per-object model named
<robot>_w_<object>.xml (it bakes the object into the sim as a free body, for
non-penetration / interaction-mesh) and that file must already exist. The repo
only ships 'largebox'. Rather than hand-author one per object, this generates
them on demand from models/<object>/<object>.obj.

Usage:
  python make_wuji_model.py                  # -> models/g1/g1_29dof_wuji.xml  (no object)
  python make_wuji_model.py largebox         # -> models/g1/g1_29dof_wuji_w_largebox.xml
  python make_wuji_model.py smallbox         # -> ..._w_smallbox.xml  (any object with a mesh)

Hand mount is derived from real hardware (Unitree G1 inspire-hand URDF flange +
the Wuji Direct-Adapter drawing), not eyeballed -- see the mount constants below.
"""
import os
import sys
import warnings
from pathlib import Path

import mujoco

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent  # scripts live in wuji/, repo root is one up
G1DIR = REPO / "models/g1"
ASSETS = str(G1DIR / "assets") + "/"
# External dep, only needed to (re)generate the models -- the built
# g1_29dof_wuji*.xml under models/g1/ already work without it. Clone
# https://github.com/wuji-technology/wuji-hand-description and point
# $WUJI_HAND_DESCRIPTION at it (default: ~/wuji-hand-description).
WUJI = Path(os.environ.get("WUJI_HAND_DESCRIPTION", Path.home() / "wuji-hand-description"))

# ── Real-hardware mount (replaces the old eyeballed QUAT/PX) ──────────────────────
# Sources: (1) Unitree G1 inspire-hand URDF -- the flange frame Unitree bolts ANY hand
# to on wrist_yaw_link; (2) the Wuji "Direct-Adapter assembled" drawing + docking STL --
# the adapter puck between the flange and the palm.
#   FLANGE_X  0.0415  = Unitree official flange offset (our old 0.035 was 6.5mm short).
#   ADAPTER_STANDOFF 0.026 = Direct-Adapter net standoff (drawing "26+-0.1"); the old
#                            model OMITTED the adapter, so the hand sat ~3cm too close.
# MOUNT_X is the palm mounting-face distance from the wrist along the arm's forward (+x) axis.
# Rotation: the wuji palm's +z (fingers) must point along +x (forward); QUAT does that, and
# left/right are MIRRORED (Unitree uses different rpy per side). The CLOCKING (thumb direction
# about the mount axis) is the one thing a 2D drawing can't pin -- certify it by rendering the
# built model against wuji-hand&Direct-Adapter-assembled-v1.pdf and adjust QUAT_* if twisted.
FLANGE_X = 0.0415
ADAPTER_STANDOFF = 0.026
MOUNT_X = FLANGE_X + ADAPTER_STANDOFF
# The wuji left.xml / right.xml are ALREADY mirror models, so the SAME mount quaternion
# gives palm-down on both hands (Unitree's per-side rpy is for ONE inspire model mounted
# mirrored -- copying it here double-flips the left hand, palm-up). Same quat for both.
QUAT_R = QUAT_L = [0.7071068, 0.0, 0.7071068, 0.0]   # palm +z (fingers) -> arm +x, palm down


def make(obj_name=None, px=MOUNT_X, weld=False):
    if not (WUJI / "mjcf/left.xml").is_file():
        sys.exit(f"Wuji hand description not found at {WUJI}. Clone wuji-hand-description and set "
                 f"$WUJI_HAND_DESCRIPTION. (The pre-built models in {G1DIR} already work without it.)")
    g1 = mujoco.MjSpec.from_file(str(G1DIR / "g1_29dof.xml"))
    wl = mujoco.MjSpec.from_file(str(WUJI / "mjcf/left.xml"))
    wr = mujoco.MjSpec.from_file(str(WUJI / "mjcf/right.xml"))
    for s in (g1, wl, wr):
        s.meshdir = ASSETS
    g1.delete(g1.body("left_rubber_hand_link"))
    g1.delete(g1.body("right_rubber_hand_link"))
    fl = g1.body("left_wrist_yaw_link").add_frame();  fl.pos = [px, 0, 0]; fl.quat = QUAT_L
    fr = g1.body("right_wrist_yaw_link").add_frame(); fr.pos = [px, 0, 0]; fr.quat = QUAT_R
    g1.attach(wl, prefix="wj_", frame=fl)
    g1.attach(wr, prefix="wjr_", frame=fr)

    if weld:
        # Phase 1: weld the fingers (delete their actuators + joints -> rigid at
        # open pose, so robot_dof stays 29) and make finger geoms non-colliding
        # (the hand retargeter owns finger-object contact). The palm stays
        # colliding so the body/wrist still respects the box in OmniRetarget.
        for a in list(g1.actuators):
            if "finger" in a.name:
                g1.delete(a)
        for j in list(g1.joints):
            if "finger" in j.name:
                g1.delete(j)
        for gm in g1.geoms:
            if gm.parent is not None and "finger" in gm.parent.name:
                gm.contype = 0
                gm.conaffinity = 0

    if obj_name:
        mesh = REPO / f"models/{obj_name}/{obj_name}.obj"
        if not mesh.is_file():
            sys.exit(f"object mesh not found: {mesh}")
        g1.add_mesh(name=f"{obj_name}_mesh", file=str(mesh))
        b = g1.worldbody.add_body(name=f"{obj_name}_link", pos=[0.5, 0.0, 0.3])
        b.add_freejoint()
        gm = b.add_geom()
        gm.type = mujoco.mjtGeom.mjGEOM_MESH
        gm.meshname = f"{obj_name}_mesh"
        gm.rgba = [0.8, 0.5, 0.3, 1.0]
        # Visual only: the retargeter uses sampled object POINTS for the
        # interaction mesh, not this sim geom. Make it non-colliding so it
        # doesn't add collision/ground constraints that break the SQP.
        gm.contype = 0
        gm.conaffinity = 0

    g1.compile()
    tag = ("_welded" if weld else "") + (("_w_" + obj_name) if obj_name else "")
    out = G1DIR / f"g1_29dof_wuji{tag}.xml"
    out.write_text(g1.to_xml())
    m = mujoco.MjModel.from_xml_path(str(out))
    print(f"[make_wuji_model] wrote {out.name}: nq={m.nq} nv={m.nv}"
          + ("  (fingers welded, finger geoms non-colliding)" if weld else "")
          + (f"  (+object '{obj_name}')" if obj_name else ""))
    return out


if __name__ == "__main__":
    weld = "--weld" in sys.argv
    pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    arg = pos[0] if pos and pos[0] not in ("none", "-") else None
    make(arg, float(pos[1]) if len(pos) > 1 else MOUNT_X, weld=weld)
