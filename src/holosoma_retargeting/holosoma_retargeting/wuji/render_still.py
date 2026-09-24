#!/usr/bin/env python
"""Offscreen still render of a model, framed on one hand, for MOUNT certification.

The interactive MuJoCo viewer needs a display (X11); this renders headless (same as
render_mp4.py) so you get a PNG you can open next to the adapter drawing. It aims the
camera at a chosen body (default the right palm) and renders a few angles into one image
so you can check how the hand sits on the wrist -- flange offset, standoff, thumb clocking.

Usage:
    python wuji/render_still.py <model.xml> [out.png] [--body wjr_right_palm_link]
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")                # headless GPU render (same as render_mp4.py)
import sys
import numpy as np
import mujoco
from PIL import Image

model_xml = sys.argv[1]
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else (
    sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "renders/mount.png")
body = sys.argv[sys.argv.index("--body") + 1] if "--body" in sys.argv else "wjr_right_palm_link"

m = mujoco.MjModel.from_xml_path(model_xml)
d = mujoco.MjData(m)

tpose = "--tpose" in sys.argv
if tpose:                                                # raise both arms out to the sides
    for side, sgn in (("left", 1.0), ("right", -1.0)):   # roll sign is mirrored per arm
        for jn, val in ((f"{side}_shoulder_pitch_joint", 0.0), (f"{side}_shoulder_roll_joint", sgn * 1.3),
                        (f"{side}_shoulder_yaw_joint", 0.0), (f"{side}_elbow_joint", 0.0)):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if j >= 0:
                d.qpos[m.jnt_qposadr[j]] = val
mujoco.mj_forward(m, d)                                   # (fingers open)

if tpose:                                                # frame the WHOLE robot (auto-fit bbox, legs incl.)
    P = d.xpos[1:]                                        # all body world positions (skip worldbody)
    look = (P.min(0) + P.max(0)) / 2.0                   # center of the robot's bounding box
    span = float(np.max(P.max(0) - P.min(0)))            # largest extent (arm span or height)
    views = [("front", 90, -8), ("top", 90, -80), ("front-low", 90, 6), ("back", -90, -8)]
    default_dist = span * 1.9                            # zoom out to guarantee feet+hands fit
else:                                                    # close-up on one hand
    look = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)].copy()
    views = [("side", 90, -15), ("front-45", 35, -20), ("top", 90, -70), ("thumb-side", 200, -15)]
    default_dist = 0.33
dist = float(sys.argv[sys.argv.index("--dist") + 1]) if "--dist" in sys.argv else default_dist  # camera range (m)
W = H = 460                                              # <=480: fits MuJoCo's default offscreen buffer
import os
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

if out.endswith(".mp4"):
    # TURNTABLE: orbit the camera 360 deg around the (posed) robot -> mp4. Pair with --tpose
    # to spin a T-pose so both hands are visible from every angle.
    import imageio
    frames = []
    with mujoco.Renderer(m, H, W) as r:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = look
        cam.distance = dist
        cam.elevation = -12
        for az in range(0, 360, 3):                      # 120 frames, 3 deg apart
            cam.azimuth = az
            r.update_scene(d, camera=cam)
            frames.append(r.render().copy())
    imageio.mimsave(out, frames, fps=30, quality=8, macro_block_size=1)
    print(f"wrote {out}  (turntable, tpose={tpose})")
else:
    # STILL: the fixed viewpoints stitched into one labelled strip
    from PIL import ImageDraw
    pad = 4
    tiles = []
    with mujoco.Renderer(m, H, W) as r:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = look
        cam.distance = dist                              # framing range (tune with --dist)
        for name, az, el in views:
            cam.azimuth, cam.elevation = az, el
            r.update_scene(d, camera=cam)
            tiles.append((name, Image.fromarray(r.render().copy())))
    strip = Image.new("RGB", (W * len(tiles) + pad * (len(tiles) - 1), H + 22), (255, 255, 255))
    for i, (name, img) in enumerate(tiles):
        strip.paste(img, (i * (W + pad), 22))
        ImageDraw.Draw(strip).text((i * (W + pad) + 6, 5), name, fill=(0, 0, 0))
    strip.save(out)
    print(f"wrote {out}  (body={body}, views={[v[0] for v in views]})")
