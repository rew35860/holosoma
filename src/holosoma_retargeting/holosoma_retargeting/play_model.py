#!/usr/bin/env python
"""Static viser view of any MuJoCo model (default pose). Inspect the Wuji+G1 graft.

Run:  python play_model.py [path_to_xml] [port]
      (default: models/g1/g1_29dof_wuji.xml, port 8081)
Then open http://localhost:<port>
"""
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import trimesh
import viser

REPO = Path(__file__).resolve().parent
XML = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "models/g1/g1_29dof_wuji.xml"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8081

m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
# Stand on the floor: lift the free base so the lowest geom sits at z=0
# (default qpos puts the base at the origin, sinking the legs below the grid).
if m.nq >= 3:
    d.qpos[2] -= float(d.geom_xpos[:, 2].min())
    mujoco.mj_forward(m, d)
print(f"[play_model] {XML.name}: nq={m.nq} nv={m.nv} ngeom={m.ngeom}")


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
n = 0
for gid in range(m.ngeom):
    lm = geom_mesh(gid)
    if lm is None:
        continue
    v, f = lm
    rgba = m.geom_rgba[gid]
    color = tuple(int(255*c) for c in rgba[:3]) if rgba[3] > 0 else (170, 170, 180)
    # Wuji geoms (mesh dataid -> name starting wj_ / belonging to wuji bodies) tinted blue-ish via their own rgba already.
    pos = d.geom_xpos[gid].copy()
    R = d.geom_xmat[gid].reshape(3, 3)
    fr = server.scene.add_frame(f"/m/g{gid}", show_axes=False, position=pos, wxyz=mat2wxyz(R))
    server.scene.add_mesh_simple(f"/m/g{gid}/s", vertices=v, faces=f, color=color, flat_shading=True)
    n += 1
print(f"[play_model] rendered {n} geoms. open http://localhost:{PORT}  (Ctrl-C to stop)")
while True:
    time.sleep(1.0)
