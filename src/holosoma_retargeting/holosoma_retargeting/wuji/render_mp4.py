#!/usr/bin/env python
"""Offscreen-render a Wuji qpos trajectory to an mp4 (shows fingers + box, so
finger/box penetration is visible -- unlike the 29-DOF viser_deploy server).

Usage: python render_mp4.py <npz> <out.mp4> [model.xml] [--track] [--az DEG] [--el DEG] [--dist METERS] [--no-floor]
  --track : camera follows the box each frame (close-up of the grasp)
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # headless GPU render
import numpy as np
import mujoco
import imageio

HOLO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # wuji/ -> repo root
G1 = HOLO + "/models/g1"


def add_default_floor(spec):
    """Add MuJoCo's stock checkerboard floor + gradient skybox to the spec.

    The hand-only scene has no ground and renders on a black void, so you can't tell whether the table
    is moving or the fingers are sliding. A z=0 floor (the scene is floor-normalized) gives that
    reference. No-ops on any element the model already declares. Floor is contype/conaffinity 0 --
    visual only, never touches the solve or contacts.
    """
    tnames, mnames, gnames = ({e.name for e in s} for s in (spec.textures, spec.materials, spec.geoms))
    if not any(t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX for t in spec.textures):
        t = spec.add_texture(); t.name = "skybox"; t.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
        t.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
        t.rgb1 = [.3, .5, .7]; t.rgb2 = [0, 0, 0]; t.width = 512; t.height = 512
    # If the model ALREADY has a ground plane (e.g. 'ground', not 'floor'), do NOT add our own -- two
    # coincident z=0 planes z-fight into shards ("breaking glass"). Keep the model's own (glossy) floor.
    if any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.geoms):
        return
    if "grid" not in tnames:
        t = spec.add_texture(); t.name = "grid"; t.type = mujoco.mjtTexture.mjTEXTURE_2D
        t.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
        t.rgb1 = [.2, .3, .4]; t.rgb2 = [.3, .4, .5]; t.width = 300; t.height = 300
    if "grid" not in mnames:
        mat = spec.add_material(); mat.name = "grid"
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
        mat.texrepeat = [6, 6]; mat.texuniform = True; mat.reflectance = 0.1
    if "floor" not in gnames:
        g = spec.worldbody.add_geom(); g.name = "floor"; g.type = mujoco.mjtGeom.mjGEOM_PLANE
        g.size = [0, 0, 0.05]; g.material = "grid"; g.contype = 0; g.conaffinity = 0

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("npz", help="trajectory containing a qpos array")
parser.add_argument("out", help="output MP4 path")
parser.add_argument("model_xml", nargs="?", help="MuJoCo scene XML; inferred from qpos width if omitted")
parser.add_argument("--track", action="store_true", help="follow the scene centroid")
parser.add_argument("--ghost", action="store_true", help="make the object translucent")
parser.add_argument("--no-floor", action="store_true", help="disable the added visual floor")
parser.add_argument("--az", type=float, default=90, help="camera azimuth in degrees")
parser.add_argument("--el", type=float, default=-18, help="camera elevation in degrees")
parser.add_argument("--dist", type=float, help="fixed camera distance in metres")
args = parser.parse_args()
npz, out = args.npz, args.out
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
q = np.load(npz, allow_pickle=True)["qpos"]
xml = args.model_xml or (G1 + "/g1_29dof_wuji_w_largebox.xml" if q.shape[1] >= 83 else G1 + "/g1_29dof_wuji.xml")

W, H = 1024, 768
_spec = mujoco.MjSpec.from_file(xml)
_spec.visual.global_.offwidth = W      # default offscreen buffer is 640x480
_spec.visual.global_.offheight = H
# object name parsed from the scene xml (…_w_<obj>.xml) — used for --ghost only
obj_name = os.path.basename(xml).split("_w_")[-1].rsplit(".xml", 1)[0] if "_w_" in xml else ""
if args.ghost and obj_name:  # translucent object -> buried fingers visible
    for gm in _spec.geoms:
        if gm.parent is not None and obj_name in (gm.parent.name or ""):
            gm.rgba = [0.85, 0.55, 0.35, 0.4]
if not args.no_floor:
    add_default_floor(_spec)               # ground + skybox so motion reads against a reference
m = _spec.compile()
d = mujoco.MjData(m)
assert q.shape[1] == m.nq, f"qpos width {q.shape[1]} != model nq {m.nq}"
r = mujoco.Renderer(m, H, W)
cam = mujoco.MjvCamera()

# Auto-frame the whole scene (robot + object) — no hardcoded object name, so this
# works for any object. lookat = per-frame scene centroid; distance fits the spread.
centers = np.empty((q.shape[0], 3))
for t in range(q.shape[0]):
    d.qpos[:] = q[t]
    mujoco.mj_forward(m, d)
    centers[t] = d.xpos[1:].mean(0)    # mean of all non-world bodies
span = float(np.linalg.norm(centers.max(0) - centers.min(0)))
if args.dist is not None:
    # fixed-zoom mode: look at the MEDIAN centroid (robust to non-contact drift frames that fling the
    # hand metres away and would otherwise pull the auto-frame lookat up and zoom the camera out)
    cam.lookat[:] = np.median(centers, axis=0)
    cam.distance = args.dist
else:
    cam.lookat[:] = centers.mean(0)
    cam.distance = 2.0 if args.track else max(2.4, span * 1.3)   # was 0.7/1.6 — too tight
cam.azimuth = args.az
cam.elevation = args.el

frames = []
for t in range(q.shape[0]):
    d.qpos[:] = q[t]
    mujoco.mj_forward(m, d)
    if args.track:
        cam.lookat[:] = centers[t]     # follow the scene centroid
    r.update_scene(d, cam)
    frames.append(r.render())

imageio.imwrite(out.rsplit(".", 1)[0] + "_sample.png", frames[len(frames) // 2])
imageio.mimsave(out, frames, fps=30, quality=8, macro_block_size=1)
print(f"[render] {len(frames)} frames -> {out}  (sample png alongside)")

# The mp4 is written; EGL/GL context teardown on normal exit spams harmless
# "EGLError" tracebacks, so exit hard to skip those destructors.
sys.stdout.flush()
os._exit(0)
