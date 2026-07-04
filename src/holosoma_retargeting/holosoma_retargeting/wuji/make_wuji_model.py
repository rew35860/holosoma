#!/usr/bin/env python
"""Generate the G1 + Wuji (both hands) MuJoCo model, optionally with an object baked in.

OmniRetarget loads a per-object model <robot>_w_<object>.xml (object as a free body); the
repo only ships 'largebox', so this generates the rest from models/<object>/<object>.obj.

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
# Only needed to (re)generate models; the built g1_29dof_wuji*.xml already work without it.
# Clone wuji-hand-description and set $WUJI_HAND_DESCRIPTION (default: ~/wuji-hand-description).
WUJI = Path(os.environ.get("WUJI_HAND_DESCRIPTION", Path.home() / "wuji-hand-description"))

# Real-hardware mount: wrist_yaw_link -> flange -> adapter -> palm.
#   FLANGE_X 0.0415        G1 flange offset (Unitree G1 + inspire-hand URDF).
#   ADAPTER_STANDOFF 0.026 Wuji Direct-Adapter puck (drawing "26+-0.1").
# QUAT sends the palm's +z (fingers) to +x (arm forward), palm down. left.xml/right.xml are
# already mirrored, so the SAME quat works for both. Thumb clocking: certify by rendering vs
# wuji-hand&Direct-Adapter-assembled-v1.pdf and tweak QUAT if twisted.
FLANGE_X = 0.0415
ADAPTER_STANDOFF = 0.026
MOUNT_X = FLANGE_X + ADAPTER_STANDOFF                 # palm distance from wrist along +x
QUAT_R = QUAT_L = [0.7071068, 0.0, 0.7071068, 0.0]


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
        # Weld fingers (drop their actuators+joints -> rigid open, robot_dof stays 29) and make
        # finger geoms non-colliding (the retargeter owns finger contact); palm keeps colliding.
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
        # Visual only (the retargeter samples object POINTS, not this geom); non-colliding
        # so it adds no collision/ground constraints that would break the SQP.
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
