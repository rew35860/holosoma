#!/usr/bin/env python
"""Offscreen-render a Wuji qpos trajectory to an mp4 (shows fingers + box, so
finger/box penetration is visible -- unlike the 29-DOF viser_deploy server).

Usage: python render_mp4.py <npz> <out.mp4> [model.xml] [--track]
  --track : camera follows the box each frame (close-up of the grasp)
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # headless GPU render
import numpy as np
import mujoco
import imageio

HOLO = os.path.dirname(os.path.abspath(__file__))
G1 = HOLO + "/models/g1"

npz = sys.argv[1]
out = sys.argv[2]
track = "--track" in sys.argv
pos = [a for a in sys.argv[3:] if not a.startswith("-")]
q = np.load(npz, allow_pickle=True)["qpos"]
xml = pos[0] if pos else (G1 + "/g1_29dof_wuji_w_largebox.xml" if q.shape[1] >= 83 else G1 + "/g1_29dof_wuji.xml")

W, H = 1024, 768
_spec = mujoco.MjSpec.from_file(xml)
_spec.visual.global_.offwidth = W      # default offscreen buffer is 640x480
_spec.visual.global_.offheight = H
if "--ghost" in sys.argv:              # translucent box -> buried fingers visible
    for gm in _spec.geoms:
        if gm.parent is not None and "largebox" in (gm.parent.name or ""):
            gm.rgba = [0.85, 0.55, 0.35, 0.4]
m = _spec.compile()
d = mujoco.MjData(m)
assert q.shape[1] == m.nq, f"qpos width {q.shape[1]} != model nq {m.nq}"
r = mujoco.Renderer(m, H, W)
cam = mujoco.MjvCamera()
box_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "largebox_link")

# precompute box path for camera target
boxpos = []
for t in range(q.shape[0]):
    d.qpos[:] = q[t]
    mujoco.mj_forward(m, d)
    boxpos.append(d.xpos[box_b].copy() if box_b >= 0 else d.xpos[1].copy())
boxpos = np.array(boxpos)
cam.lookat[:] = boxpos.mean(0)
cam.distance = 0.7 if track else 1.6
cam.azimuth = 135
cam.elevation = -18

frames = []
for t in range(q.shape[0]):
    d.qpos[:] = q[t]
    mujoco.mj_forward(m, d)
    if track:
        cam.lookat[:] = boxpos[t]
    r.update_scene(d, cam)
    frames.append(r.render())

imageio.imwrite(out.rsplit(".", 1)[0] + "_sample.png", frames[len(frames) // 2])
imageio.mimsave(out, frames, fps=30, quality=8, macro_block_size=1)
print(f"[render] {len(frames)} frames -> {out}  (sample png alongside)")
