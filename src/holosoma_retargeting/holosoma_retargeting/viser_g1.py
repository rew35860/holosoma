"""Interactive viser viewer for a retargeted G1 + object trajectory — the
"after OmniRetarget" stage. Reads the retargeter's .npz (qpos) and the MuJoCo
G1+object model, plays back in a browser at http://localhost:8080 with a
pause/unpause button + frame scrubber.

Faithful to the MP4: it drives the SAME MuJoCo model with the SAME qpos, then
reads each mesh geom's world transform per frame and updates viser nodes.

Usage (from the holosoma_retargeting working dir, hsretargeting env):
    python viser_g1.py <task_name> <object_name> [save_dir] [--port 8080]

Example:
    python viser_g1.py sub14_suitcase_001 suitcase \
        demo_results/g1/object_interaction/omomo_test

viser >= 1.0.29: mesh handles support live .position / .wxyz updates, so each
mesh geom is added ONCE and only its transform is updated per frame (efficient).
"""
import os
import sys
import time
import argparse

import numpy as np
import mujoco
import viser
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser()
parser.add_argument("task", help="task name (npz basename without _original)")
parser.add_argument("object", help="object name (for the XML)")
parser.add_argument("save_dir", nargs="?",
                    default="demo_results/g1/object_interaction/omomo_test",
                    help="dir holding <task>_original.npz")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--fps", type=int, default=0, help="override fps (0 = use npz fps)")
args = parser.parse_args()

XML = f"models/g1/g1_29dof_w_{args.object}.xml"
NPZ = f"{args.save_dir}/{args.task}_original.npz"

# ── load model + trajectory ──
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
data = np.load(NPZ)
qpos = data["qpos"]
T = qpos.shape[0]
fps = args.fps or (int(data["fps"]) if "fps" in data and data["fps"].size else 30)
cost = float(data["cost"]) if "cost" in data and data["cost"].size else float("nan")
print(f"{args.task}: qpos {qpos.shape}, fps {fps}, cost {cost:.3f}")

# ── extract every MESH geom's local geometry once (robot links + object) ──
# Collision spheres/cylinders are skipped for a clean visual.
geoms = []   # list of dict(gid, verts, faces, color)
for g in range(m.ngeom):
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
    mid = m.geom_dataid[g]
    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
    verts = m.mesh_vert[va:va + vn].reshape(-1, 3).astype(np.float32)
    faces = m.mesh_face[fa:fa + fn].reshape(-1, 3).astype(np.int32)
    rgba = m.geom_rgba[g]
    color = tuple(int(255 * c) for c in rgba[:3]) if rgba[:3].sum() > 0 else (180, 180, 185)
    geoms.append(dict(gid=g, verts=verts, faces=faces, color=color))
print(f"  {len(geoms)} mesh geoms (robot links + object)")

# ── viser scene (G1 world is z-up) ──
server = viser.ViserServer(port=args.port)
server.scene.set_up_direction("+z")
server.scene.add_grid("/floor", width=10, height=10, plane="xy")

nodes = []
for i, gm in enumerate(geoms):
    nodes.append(server.scene.add_mesh_simple(
        f"/geom/{gm['gid']}", vertices=gm["verts"], faces=gm["faces"],
        color=gm["color"], flat_shading=True,
    ))

def pose_frame(t: int):
    """Set qpos, run FK, update each viser node's world position + orientation."""
    d.qpos[:] = qpos[t]
    mujoco.mj_forward(m, d)
    for node, gm in zip(nodes, geoms):
        g = gm["gid"]
        node.position = d.geom_xpos[g].copy()
        quat_xyzw = Rotation.from_matrix(d.geom_xmat[g].reshape(3, 3)).as_quat()
        node.wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

# ── GUI: pause/unpause + scrubber + fps + restart ──
state = {"playing": True, "frame": 0}
with server.gui.add_folder("playback"):
    btn_toggle  = server.gui.add_button("⏸  Pause")
    gui_frame   = server.gui.add_slider("frame", min=0, max=T - 1, step=1, initial_value=0)
    gui_fps     = server.gui.add_slider("fps", min=1, max=60, step=1, initial_value=fps)
    btn_restart = server.gui.add_button("⏮  Restart")
status = server.gui.add_markdown("**▶ Playing**")
server.gui.add_markdown(f"_{args.task}_  ·  cost {cost:.3f}")

@btn_toggle.on_click
def _(_):
    state["playing"] = not state["playing"]
    btn_toggle.label = "▶  Play" if not state["playing"] else "⏸  Pause"
    status.content = "**⏸ Paused**" if not state["playing"] else "**▶ Playing**"

@btn_restart.on_click
def _(_):
    state["frame"] = 0
    gui_frame.value = 0
    pose_frame(0)

@gui_frame.on_update
def _(_):
    state["frame"] = int(gui_frame.value)
    pose_frame(state["frame"])

pose_frame(0)
print(f"\nviser running at http://localhost:{args.port}   (Ctrl+C to stop)")
print(f"  {T} frames")

while True:
    if state["playing"]:
        state["frame"] = (state["frame"] + 1) % T
        gui_frame.value = state["frame"]
    time.sleep(1.0 / max(1, int(gui_fps.value)))
