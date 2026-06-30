#!/usr/bin/env python
"""Viser playback of the Wuji-hands retarget result (both hands, 10 fingertips).

Renders the G1+Wuji model (from the combined MJCF) for the saved qpos, and
overlays the 10 human fingertips (RED) vs the 10 robot fingertips (ORANGE).

Run:  python play_wuji.py [npz] [port]
Then open http://localhost:<port>  (default 8082)
"""
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import trimesh
import viser

REPO = Path(__file__).resolve().parent.parent  # scripts live in wuji/, repo root is one up
NPZ = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "demo_results_wuji/sub3_largebox_003.npz"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8082
MODEL_ARG = sys.argv[3] if len(sys.argv) > 3 else None  # optional explicit model xml

from holosoma_retargeting.config_types import data_type as dt
J = dt.SMPLH_DEMO_JOINTS
# (human joint, robot tip body) pairs, both hands
PAIRS = [
    ("L_Thumb3", "wj_left_finger1_link4"), ("L_Index3", "wj_left_finger2_link4"),
    ("L_Middle3", "wj_left_finger3_link4"), ("L_Ring3", "wj_left_finger4_link4"),
    ("L_Pinky3", "wj_left_finger5_link4"),
    ("R_Thumb3", "wjr_right_finger1_link4"), ("R_Index3", "wjr_right_finger2_link4"),
    ("R_Middle3", "wjr_right_finger3_link4"), ("R_Ring3", "wjr_right_finger4_link4"),
    ("R_Pinky3", "wjr_right_finger5_link4"),
]
hum_idx = [J.index(h) for h, _ in PAIRS]

data = np.load(NPZ, allow_pickle=True)
q = np.asarray(data["qpos"], float)
hj = np.asarray(data["human_joints"], float)
fps = int(data["fps"]) if "fps" in data.files else 30
T = q.shape[0]
print(f"[play_wuji] {NPZ.name}: frames={T} qpos={q.shape}")

# Match the model to the result: object-interaction qpos includes the object's
# 7 free-joint DOF (nq 83 vs 76), so load the _w_largebox model to show the box.
if MODEL_ARG:
    XML = Path(MODEL_ARG).resolve()
elif q.shape[1] >= 83:
    XML = REPO / "models/g1/g1_29dof_wuji_w_largebox.xml"
else:
    XML = REPO / "models/g1/g1_29dof_wuji.xml"
print(f"[play_wuji] using model {XML.name}")

m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
rob_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for _, b in PAIRS]


def mat2wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
    if i == 0:
        s = np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    if i == 1:
        s = np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    s = np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2
    return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def geom_mesh(gid):
    gt = m.geom_type[gid]; size = m.geom_size[gid]
    if gt == mujoco.mjtGeom.mjGEOM_MESH:
        did = m.geom_dataid[gid]
        va, vn = m.mesh_vertadr[did], m.mesh_vertnum[did]
        fa, fn = m.mesh_faceadr[did], m.mesh_facenum[did]
        return (m.mesh_vert[va:va+vn].reshape(-1,3).astype(np.float32),
                m.mesh_face[fa:fa+fn].reshape(-1,3).astype(np.int32))
    try:
        if gt == mujoco.mjtGeom.mjGEOM_SPHERE: ms = trimesh.creation.icosphere(radius=float(size[0]), subdivisions=1)
        elif gt == mujoco.mjtGeom.mjGEOM_BOX: ms = trimesh.creation.box(extents=2*size[:3])
        elif gt == mujoco.mjtGeom.mjGEOM_CAPSULE: ms = trimesh.creation.capsule(radius=float(size[0]), height=float(2*size[1]))
        elif gt == mujoco.mjtGeom.mjGEOM_CYLINDER: ms = trimesh.creation.cylinder(radius=float(size[0]), height=float(2*size[1]))
        else: return None
        return np.asarray(ms.vertices, np.float32), np.asarray(ms.faces, np.int32)
    except Exception:
        return None


server = viser.ViserServer(port=PORT)
server.scene.add_grid("/grid", width=4, height=4)
handles = []
for gid in range(m.ngeom):
    lm = geom_mesh(gid)
    if lm is None:
        continue
    v, f = lm
    rgba = m.geom_rgba[gid]
    color = tuple(int(255*c) for c in rgba[:3]) if rgba[3] > 0 else (170, 170, 180)
    fr = server.scene.add_frame(f"/r/g{gid}", show_axes=False)
    server.scene.add_mesh_simple(f"/r/g{gid}/s", vertices=v, faces=f, color=color, flat_shading=True)
    handles.append((gid, fr))


def pts(name, n, color, size):
    return server.scene.add_point_cloud(name, points=np.zeros((n, 3), np.float32),
                                        colors=np.tile(np.array(color, np.uint8), (n, 1)), point_size=size)


human_fingers = pts("/human_tips", 10, (255, 60, 60), 0.02)   # RED = human fingertips
robot_fingers = pts("/robot_tips", 10, (255, 160, 0), 0.02)   # ORANGE = robot fingertips
frame = server.gui.add_slider("Frame", min=0, max=T - 1, step=1, initial_value=0)
play = server.gui.add_checkbox("Play", True)
speed = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
server.gui.add_markdown("RED = human fingertips · ORANGE = robot (Wuji) fingertips")


def show(i):
    d.qpos[:] = q[i]
    mujoco.mj_forward(m, d)
    for gid, fr in handles:
        fr.position = d.geom_xpos[gid].copy()
        fr.wxyz = mat2wxyz(d.geom_xmat[gid].reshape(3, 3))
    human_fingers.points = hj[i, hum_idx].astype(np.float32)
    robot_fingers.points = np.array([d.xpos[b] for b in rob_bodies], np.float32)


@frame.on_update
def _(_):
    if not play.value:
        show(int(frame.value))


show(0)
print(f"[play_wuji] rendered {len(handles)} geoms. open http://localhost:{PORT}  (Ctrl-C to stop)")
i = 0.0
last = time.perf_counter()
while True:
    now = time.perf_counter()
    if play.value:
        i = (i + (now - last) * fps * speed.value) % T
        frame.value = int(i)
        show(int(i))
    last = now
    time.sleep(1.0 / max(1, fps))
