#!/usr/bin/env python
"""Viser playback of the dexterous-finger retarget result (no URDF needed).

Renders the retargeted G1 (from the 35-DOF MuJoCo model) frame-by-frame and
overlays the human thumb/pinky tips + the robot thumb/pinky links, so you can
see the fingers tracking. Replays the SAVED npz (instant, no re-solve).

Run:  python play_fingers.py   (in the omniretarget env)
Then open the printed http://localhost:8080 URL in a browser.
"""
import time
from pathlib import Path

import numpy as np
import mujoco
import trimesh
import viser

REPO = Path(__file__).resolve().parent
NPZ = REPO / "demo_results_fingers/sub3_largebox_003.npz"
XML = REPO / "models/g1/g1_29dof_dexfinger.xml"

# --- finger joint indices in the human (SMPLH) joint array ---
from holosoma_retargeting.config_types import data_type as dt
J = dt.SMPLH_DEMO_JOINTS
iThumb, iPinky, iWrist = J.index("L_Thumb3"), J.index("L_Pinky3"), J.index("L_Wrist")

data = np.load(NPZ, allow_pickle=True)
q = np.asarray(data["qpos"], dtype=np.float64)        # (T, 42)
hj = np.asarray(data["human_joints"], dtype=np.float64)  # (T, 52, 3)
fps = int(data["fps"]) if "fps" in data.files else 30
T = q.shape[0]
print(f"[play_fingers] frames={T} fps={fps} qpos={q.shape} human_joints={hj.shape}")

m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
bThumb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_thumb_link")
bPinky = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_pinky_link")


def mat2wxyz(R):
    """3x3 rotation matrix -> (w,x,y,z) quaternion."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return np.array([w, x, y, z])


def geom_local_mesh(gid):
    gt = m.geom_type[gid]
    size = m.geom_size[gid]
    if gt == mujoco.mjtGeom.mjGEOM_MESH:
        did = m.geom_dataid[gid]
        va, vn = m.mesh_vertadr[did], m.mesh_vertnum[did]
        fa, fn = m.mesh_faceadr[did], m.mesh_facenum[did]
        v = m.mesh_vert[va:va + vn].reshape(-1, 3).astype(np.float32)
        f = m.mesh_face[fa:fa + fn].reshape(-1, 3).astype(np.int32)
        return v, f
    try:
        if gt == mujoco.mjtGeom.mjGEOM_SPHERE:
            ms = trimesh.creation.icosphere(radius=float(size[0]), subdivisions=1)
        elif gt == mujoco.mjtGeom.mjGEOM_BOX:
            ms = trimesh.creation.box(extents=(2 * size[:3]))
        elif gt == mujoco.mjtGeom.mjGEOM_CAPSULE:
            ms = trimesh.creation.capsule(radius=float(size[0]), height=float(2 * size[1]))
        elif gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
            ms = trimesh.creation.cylinder(radius=float(size[0]), height=float(2 * size[1]))
        else:
            return None
        return np.asarray(ms.vertices, np.float32), np.asarray(ms.faces, np.int32)
    except Exception:
        return None


server = viser.ViserServer()
server.scene.add_grid("/grid", width=4, height=4)

# Build robot geom meshes once (under per-geom frames we move each step).
handles = []
for gid in range(m.ngeom):
    lm = geom_local_mesh(gid)
    if lm is None:
        continue
    v, f = lm
    rgba = m.geom_rgba[gid]
    color = tuple(int(255 * c) for c in rgba[:3]) if rgba[3] > 0 else (170, 170, 180)
    fr = server.scene.add_frame(f"/robot/g{gid}", show_axes=False)
    server.scene.add_mesh_simple(f"/robot/g{gid}/m", vertices=v, faces=f, color=color, flat_shading=True)
    handles.append((gid, fr))
print(f"[play_fingers] rendering {len(handles)} robot geoms")


def pts(name, n, color, size):
    return server.scene.add_point_cloud(
        name, points=np.zeros((n, 3), np.float32),
        colors=np.tile(np.array(color, np.uint8), (n, 1)), point_size=size)


all_human = pts("/human/all", 52, (130, 130, 130), 0.012)      # faint full-body context
human_fingers = pts("/human/fingers", 2, (255, 60, 60), 0.03)   # RED = human thumb/pinky tips
robot_fingers = pts("/robot/fingers", 2, (255, 160, 0), 0.03)   # ORANGE = robot thumb/pinky links

frame = server.gui.add_slider("Frame", min=0, max=T - 1, step=1, initial_value=0)
play = server.gui.add_checkbox("Play", True)
speed = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
server.gui.add_markdown("RED = human thumb/pinky tip · ORANGE = robot thumb/pinky link")


def show(i):
    d.qpos[:] = q[i]
    mujoco.mj_forward(m, d)
    for gid, fr in handles:
        fr.position = d.geom_xpos[gid].copy()
        fr.wxyz = mat2wxyz(d.geom_xmat[gid].reshape(3, 3))
    all_human.points = hj[i].astype(np.float32)
    human_fingers.points = np.array([hj[i, iThumb], hj[i, iPinky]], np.float32)
    robot_fingers.points = np.array([d.xpos[bThumb], d.xpos[bPinky]], np.float32)


@frame.on_update
def _(_):
    if not play.value:
        show(int(frame.value))


show(0)
print("[play_fingers] open http://localhost:8080  (Ctrl-C to stop)")
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
