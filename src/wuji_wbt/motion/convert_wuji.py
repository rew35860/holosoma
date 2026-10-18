"""Convert a full 83-DOF Wuji retarget qpos (root7 + 69 joints + object7) into a holosoma-format WBT
motion npz that KEEPS the 40 finger joints, resampled to the RL control rate.

holosoma's MotionLoader does NOT resample: MotionCommand.step advances exactly ONE motion frame per
control step (50 Hz), so a 30-fps clip stored as-is is tracked at 1.67x speed. The qpos is therefore
resampled to `out_fps` BEFORE forward kinematics (linear for positions/hinges, slerp for the root and
object quaternions), then replayed on the g1_29dof_wuji_w_<obj> MJCF: FK gives every body pose incl. the
finger links, mj_differentiatePos gives the generalized velocities at the output dt.

Output keys match the holosoma MotionLoader: fps, joint_pos [T, 7+69], joint_vel [T, 6+69], body_pos_w,
body_quat_w (wxyz), body_lin_vel_w, body_ang_vel_w, body_names, joint_names, object_pos_w/quat_w/lin_vel_w/ang_vel_w.

Usage (omniretarget env, has mujoco):
  python convert_wuji.py <retarget_part3.npz> <g1_29dof_wuji_w_obj.xml> <out.npz> [in_fps=30] [out_fps=50]
"""
import sys
from pathlib import Path

import numpy as np
import mujoco

# reuse the stock converter's world-frame body velocity helper (no reimplementation)
RETARGET = Path(__file__).resolve().parents[3] / "holosoma_retargeting" / "holosoma_retargeting"
sys.path.insert(0, str(RETARGET.parent))
from holosoma_retargeting.data_conversion.convert_data_format_mj import world_body_velocities

JNT_FREE = 0   # mujoco joint-type code


def _slerp(q0, q1, t):
    """Batched slerp between wxyz quaternions, t in [0, 1] per row."""
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)                      # shortest arc
    d = np.abs(d).clip(max=1.0)
    th = np.arccos(d)
    s = np.sin(th)
    ok = s > 1e-6
    s_safe = np.where(ok, s, 1.0)
    w0 = np.where(ok, np.sin((1 - t) * th) / s_safe, 1 - t)
    w1 = np.where(ok, np.sin(t * th) / s_safe, t)
    out = w0 * q0 + w1 * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def resample_qpos(q, m, in_fps, out_fps):
    """Resample a qpos trajectory (T_in, nq) to out_fps; free-joint quaternion blocks are slerped."""
    T = len(q)
    n_out = int(round((T - 1) * out_fps / in_fps)) + 1
    tt = np.arange(n_out) * in_fps / out_fps            # output times in input-frame units
    i0 = np.minimum(np.floor(tt).astype(int), T - 1)
    i1 = np.minimum(i0 + 1, T - 1)
    b = (tt - i0)[:, None]
    out = (1 - b) * q[i0] + b * q[i1]                   # linear everywhere, then redo the quaternions
    for j in range(m.njnt):
        if int(m.jnt_type[j]) == JNT_FREE:
            a = int(m.jnt_qposadr[j]) + 3
            out[:, a:a + 4] = _slerp(q[i0, a:a + 4], q[i1, a:a + 4], b)
    return out


def convert(retarget_npz: str, mjcf: str, out_npz: str, in_fps: int = 30, out_fps: int = 50) -> str:
    q_in = np.load(retarget_npz, allow_pickle=True)["qpos"].astype(np.float64)   # (T, 83) = root7 + 69 + object7
    m = mujoco.MjModel.from_xml_path(mjcf)
    assert m.nq == q_in.shape[1], f"model nq {m.nq} != qpos cols {q_in.shape[1]} (need matching Wuji+object MJCF)"
    q = resample_qpos(q_in, m, in_fps, out_fps)
    T = len(q)
    d = mujoco.MjData(m)
    dt = 1.0 / out_fps
    keys = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
            "object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w")
    log = {k: [] for k in keys}
    qvel = np.zeros(m.nv)
    for t in range(T):
        d.qpos[:] = q[t]
        if t > 0:
            mujoco.mj_differentiatePos(m, qvel, dt, q[t - 1], q[t])       # tangent-space velocity (incl. quats)
        d.qvel[:] = qvel
        mujoco.mj_forward(m, d)
        lin, ang = world_body_velocities(m, d)
        log["joint_pos"].append(d.qpos[:-7].copy())          # root7 + 69 joints (object freejoint is the tail 7)
        log["joint_vel"].append(d.qvel[:-6].copy())          # root6 + 69
        log["body_pos_w"].append(d.xpos.copy())
        log["body_quat_w"].append(d.xquat.copy())            # wxyz
        log["body_lin_vel_w"].append(np.asarray(lin).copy())
        log["body_ang_vel_w"].append(np.asarray(ang).copy())
        log["object_pos_w"].append(d.qpos[-7:-4].copy())
        log["object_quat_w"].append(d.qpos[-4:].copy())      # wxyz
        log["object_lin_vel_w"].append(d.qvel[-6:-3].copy())
        log["object_ang_vel_w"].append(d.qvel[-3:].copy())
    arrays = {k: np.stack(v, axis=0) for k, v in log.items()}
    arrays["fps"] = np.array([out_fps])
    # joint_names = the 69 non-free joints (MotionLoader reindexes by NAME to the robot dof order);
    # body_names = every body (incl. finger + tip links) so hand-level terms can index them.
    jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt) if int(m.jnt_type[i]) != JNT_FREE]
    arrays["joint_names"] = np.array(jn)
    arrays["body_names"] = np.array([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)])
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **arrays)
    print(f"[convert_wuji] {len(q_in)} frames @ {in_fps}fps -> {T} frames @ {out_fps}fps ({(T - 1) / out_fps:.2f}s) "
          f"| joint_pos {arrays['joint_pos'].shape} | {len(jn)} joints | {len(arrays['body_names'])} bodies -> {out_npz}")
    return out_npz


if __name__ == "__main__":
    a = sys.argv[1:]
    convert(a[0], a[1], a[2], int(a[3]) if len(a) > 3 else 30, int(a[4]) if len(a) > 4 else 50)
