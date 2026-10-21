"""Fail-loud numerical checks for the retargeting pipeline (kept out of pipeline.py for readability).

Each function raises AssertionError with a specific message if a scale-consistency invariant is violated,
and returns any values the caller/report needs (so the main flow stays: load -> scale -> check -> save -> report).
"""
from __future__ import annotations

import numpy as np
import mujoco


def _qmat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q / (np.linalg.norm(q) + 1e-12))

    return R.reshape(3, 3)


def assert_single_scale(s: float, robot_height: float) -> float:
    """No-double-scale guard: the resolved scale must imply a RAW human height ~1.4-2.1 m
    (a scale applied twice would push it out of range). Returns the implied human_height."""
    h = robot_height / s
    assert 1.4 < h < 2.1, f"implied human_height {h:.2f}m -> suspect double-scaled s"
    return h


def assert_object_local_scaled(obj_local_s, obj_local_raw, s) -> None:
    """Object surface points must be EXACTLY s*raw (no re-/double-scale)."""
    assert np.allclose(obj_local_s, s * obj_local_raw, atol=1e-9), "object_local scaled != s*raw (double-scale)"


def object_floor_heights(vertices, poses):
    """Lowest visible mesh vertex per frame above z=0; poses are xyz + wxyz.

    Uses every vertex with its rigid pose, preserving the object's local origin.
    Does not change object poses or define a benchmark penetration metric.
    """
    vertices, poses = np.asarray(vertices), np.asarray(poses)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("Object floor check needs nonempty mesh vertices of shape (N, 3)")
    if poses.ndim != 2 or poses.shape[1] != 7 or not len(poses):
        raise ValueError("Object floor check needs nonempty poses of shape (T, 7)")
    if not np.isfinite(vertices).all() or not np.isfinite(poses).all() or np.any(np.linalg.norm(poses[:, 3:], axis=1) < 1e-9):
        raise ValueError("Object floor check received invalid vertices/poses")
    return np.array([(vertices @ _qmat(p[3:])[2]).min() + p[2] for p in poses])


def _contact_offsets(hk, obj_pos, obj_q):
    """Right-hand wrist(kp0) & mean-fingertip vertical offset above the object origin, object-local, per frame."""
    w, f = [], []
    for t in range(len(hk)):
        R = _qmat(obj_q[t]); rel = (hk[t, 1] - obj_pos[t]) @ R          # right hand, object-local
        w.append(rel[0, 2]); f.append(rel[[4, 8, 12, 16, 20], 2].mean())
    return np.array(w), np.array(f)


def assert_contact_offsets_preserved(raw_hk, hk_s, raw_obj, obj_pose_muj, s) -> dict:
    """Uniform scaling must scale the source hand<->object contact offsets by EXACTLY s.
    Returns {w_raw, w_s, f_raw, f_s, dw, df} for the report."""
    obj_pos_raw, obj_q = raw_obj[:, 4:7], raw_obj[:, :4]             # raw object translation + quaternion
    obj_pos_s = obj_pose_muj[:, :3]                                  # scaled translation (quaternion unchanged)
    w_raw, f_raw = _contact_offsets(raw_hk, obj_pos_raw, obj_q)
    w_s, f_s = _contact_offsets(hk_s, obj_pos_s, obj_q)
    dw = float(np.abs(w_s - s * w_raw).max()); df = float(np.abs(f_s - s * f_raw).max())
    assert dw < 1e-6 and df < 1e-6, f"uniform scaling did NOT preserve contact offsets (wrist {dw:.2e}, tip {df:.2e})"
    return dict(w_raw=w_raw, w_s=w_s, f_raw=f_raw, f_s=f_s, dw=dw, df=df)


def _mesh_world_z(model, data, gid):
    """World-frame z of every vertex of mesh geom `gid` (mesh scale is already baked into model.mesh_vert)."""
    mid = model.geom_dataid[gid]
    adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    v = model.mesh_vert[adr:adr + num].reshape(-1, 3)
    R = data.geom_xmat[gid].reshape(3, 3)
    return (data.geom_xpos[gid] + v @ R.T)[:, 2]


def _object_geometry(model_path, obj_name) -> dict:
    """Forward an object model at its default pose; return the object origin + visible/collision z-extents (world).
    Follows the naming convention: body `<obj>_link`, visible geom `<obj>_visual`, collision `<obj>_piece_NNN`."""
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{obj_name}_link")
    vis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{obj_name}_visual")
    assert bid >= 0 and vis >= 0, f"model {model_path} lacks '{obj_name}_link'/'{obj_name}_visual' (naming convention)"
    origin = d.xpos[bid].copy()
    vz = _mesh_world_z(m, d, vis)
    czs = []
    for i in range(64):                                             # CoACD pieces <obj>_piece_000, 001, ...
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{obj_name}_piece_{i:03d}")
        if g < 0:
            break
        czs.append(_mesh_world_z(m, d, g))
    cz = np.concatenate(czs) if czs else vz
    return dict(origin=origin, vis_top=float(vz.max()), vis_bot=float(vz.min()),
                col_top=float(cz.max()), col_bot=float(cz.min()), n_col=len(czs))


def assert_object_model_scaled(base_path, scaled_path, s, obj_name) -> dict:
    """The scaled model's object must be EXACTLY the base object x s about its origin (proving the mesh
    `scale` attr took effect), and scaling must not open a NEW visible-vs-collision gap: the CoACD top must
    stay within s x the base's own gap of the visible top (both surfaces scaled together). Returns geometry."""
    b = _object_geometry(base_path, obj_name)
    a = _object_geometry(scaled_path, obj_name)
    hb, ha = b["vis_top"] - b["origin"][2], a["vis_top"] - a["origin"][2]   # tabletop height above the origin
    assert abs(ha - s * hb) < 1e-4, f"scaled tabletop {ha:.4f} != s*base {s * hb:.4f} m (scale attr not applied)"
    vcb, vca = abs(b["vis_top"] - b["col_top"]), abs(a["vis_top"] - a["col_top"])  # visible-vs-collision gap
    assert abs(vca - s * vcb) < 1e-3, f"scaling opened a new under-fill: gap {vca * 1e3:.2f} != s*base {s * vcb * 1e3:.2f} mm"
    a.update(base_vis_top=b["vis_top"], base_col_top=b["col_top"], vc_base=vcb, vc_scaled=vca)
    return a
