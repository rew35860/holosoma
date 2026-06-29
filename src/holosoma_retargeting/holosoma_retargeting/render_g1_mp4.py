"""Render a retargeted G1 + object trajectory to MP4 from the retargeter's .npz.

Usage (from the holosoma_retargeting working dir, hsretargeting env):
    python render_g1_mp4.py <task_name> <object_name> [save_dir] [out_dir]

Example:
    python render_g1_mp4.py sub14_suitcase_001 suitcase \
        demo_results/g1/object_interaction/omomo_test ./renders_g1
"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"   # headless GPU render
import mujoco
import numpy as np
import imageio.v2 as imageio

task = sys.argv[1]
obj = sys.argv[2]
save_dir = sys.argv[3] if len(sys.argv) > 3 else "demo_results/g1/object_interaction/omomo_test"
out_dir = sys.argv[4] if len(sys.argv) > 4 else "./renders_g1"
os.makedirs(out_dir, exist_ok=True)

XML = f"models/g1/g1_29dof_w_{obj}.xml"
NPZ = f"{save_dir}/{task}_original.npz"
OUT = f"{out_dir}/retarget_{task}.mp4"

m = mujoco.MjModel.from_xml_path(XML)
# Enlarge the offscreen framebuffer so we can render above the default 640x480.
m.vis.global_.offwidth = 1280
m.vis.global_.offheight = 720
d = mujoco.MjData(m)
data = np.load(NPZ)
qpos = data["qpos"]
fps = int(data["fps"]) if "fps" in data and data["fps"].size else 30
print(f"{task}: qpos {qpos.shape}, nq {m.nq}, fps {fps}, cost {float(data['cost']):.3f}")

r = mujoco.Renderer(m, height=720, width=1280)
cam = mujoco.MjvCamera()
cam.azimuth, cam.elevation, cam.distance = 135.0, -20.0, 3.5
cam.lookat[:] = [0.0, 0.0, 0.9]

with imageio.get_writer(OUT, fps=fps) as w:
    for t in range(qpos.shape[0]):
        d.qpos[:] = qpos[t]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam)
        w.append_data(r.render())
print(f"wrote {OUT}")
