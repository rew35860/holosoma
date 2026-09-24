#!/usr/bin/env python
"""Contact-placement IK — move the ARM so the robot hand sits where the HUMAN's was (Step 2).

Step 3's finger solve produces a good CURL but the hand is mis-PLACED: Stage A parks the robot palm
8-15 cm from the human wrist (its whole-body interaction-mesh objective never pins the wrist), leaving
the fingertips 5-16 cm from the rim the human actually grabbed. The finger stage cannot fix this -- it
frees only wrist rotation + finger curl, never the hand's position -- so the arm must move.

Two formulations, selected with --target:

  wrist (default) -- match the robot PALM to the human WRIST keypoint, using ONLY shoulder+elbow
      (4 DOF). One well-posed target; the 3 wrist joints are left exactly as the finger solve set
      them, so the palm orientation it chose survives the move. Runs on ALL frames (matching the
      human wrist is valid whether or not the hand is touching), so there is no jump at contact
      boundaries. This fixes the placement error at its source.

  tips -- the first attempt, kept for comparison: drive the CURLED fingertips onto the Step-1 rim
      contacts over all 7 arm DOF. Halved the error but is structurally awkward -- a curled fingertip
      sits next to the palm, so putting it on the rim drags the palm somewhere strange, and 5 targets
      against a 6-DOF hand placement is over-determined, satisfying none of them.

Either way the FINGERS are held fixed: this is a placement, not a re-curl. Render the result to see
whether placement was the missing piece; re-run the finger solve from here to refine.

Usage (omniretarget env), run from the retargeting root:
    python wuji/place_hand.py \
        demo_results/.../<seq>.npz <seq>_contact.npz models/g1/g1_29dof_wuji_w_table_coacd.xml \
        --handkp ../../../../InterAct/result/humoto_pt/<seq>_handkp.npy [--target wrist] [--out ...]
"""
from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

# contact finger index (0=thumb..4=pinky, MediaPipe order) -> Wuji finger link (finger1=thumb..5=pinky).
# NB the model's prefixes are INCONSISTENT between sides ("wj_left_" vs "wjr_right_") -- these strings
# are verified against the model at startup (body_id) because a wrong name does not error, it silently
# indexes data.xpos[-1] and the IK chases garbage.
TIP_BODY = {"left": [f"wj_left_finger{i}_link4" for i in range(1, 6)],
            "right": [f"wjr_right_finger{i}_link4" for i in range(1, 6)]}
PALM_BODY = {"left": "wj_left_palm_link", "right": "wjr_right_palm_link"}


def body_id(model, name: str) -> int:
    """mj_name2id that refuses to fail silently (-1 would index the LAST body downstream)."""
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if b < 0:
        raise SystemExit(f"body {name!r} not in the model -- check the side prefix")
    return b
ARM_JOINTS = {s: [f"{s}_shoulder_pitch_joint", f"{s}_shoulder_roll_joint", f"{s}_shoulder_yaw_joint",
                  f"{s}_elbow_joint", f"{s}_wrist_roll_joint", f"{s}_wrist_pitch_joint",
                  f"{s}_wrist_yaw_joint"] for s in ("left", "right")}


def arm_indices(model, side: str, n_joints: int):
    """qpos/qvel addresses + limits for the first `n_joints` of one arm chain.

    n_joints=4 -> shoulder+elbow only (position placement, wrist orientation untouched);
    n_joints=7 -> the full arm (the tips formulation).

    Returns:
        tuple: (qadr, dadr, lo, hi), each (n_joints,).
    """
    qadr, dadr, lo, hi = [], [], [], []
    for n in ARM_JOINTS[side][:n_joints]:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        qadr.append(int(model.jnt_qposadr[j]))
        dadr.append(int(model.jnt_dofadr[j]))
        lo.append(float(model.jnt_range[j, 0]))
        hi.append(float(model.jnt_range[j, 1]))
    return np.array(qadr), np.array(dadr), np.array(lo), np.array(hi)


def point_jac(model, data, body_id: int, dadr: np.ndarray):
    """World position of a body origin and its positional Jacobian, arm columns only.

    Args:
        model, data: MuJoCo model/data (data must be post-mj_forward).
        body_id: the body whose origin is placed.
        dadr: dof addresses of the joints the IK may move.

    Returns:
        tuple: (pos (3,), J (3, len(dadr))).
    """
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, None, data.xpos[body_id], body_id)
    return data.xpos[body_id].copy(), jacp[:, dadr]


def ik_frame(model, data, q: np.ndarray, bodies, targets, qadr, dadr, lo, hi,
             iters: int, damp: float = 1e-3, step: float = 0.2):
    """Damped Gauss-Newton IK for one frame, in place on q's arm columns.

    Args:
        q: (nq,) full qpos for this frame (arm columns updated in place).
        bodies: body ids to place.
        targets: (n,3) world targets, one per body.
        qadr, dadr, lo, hi: from arm_indices.
        iters: iterations.
        damp: Levenberg damping (keeps steps sane when a target is unreachable).
        step: max |dq| per iteration (rad).

    Returns:
        float: final RMS body->target error (m).
    """
    r = np.zeros(3 * len(bodies))
    for _ in range(iters):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        rows, res = [], []
        for b, tgt in zip(bodies, targets):
            p, J = point_jac(model, data, b, dadr)
            rows.append(J)
            res.append(p - tgt)
        J = np.vstack(rows)
        r = np.concatenate(res)
        dq = -np.linalg.solve(J.T @ J + damp * np.eye(len(qadr)), J.T @ r)   # damped least squares
        dq = np.clip(dq, -step, step)
        q[qadr] = np.clip(q[qadr] + dq, lo, hi)
    return float(np.sqrt(np.mean(r ** 2)))


def rim_error(model, data, q, tip_ids, tgt, contact, t, h):
    """RMS distance from the contact fingers' tips to their Step-1 rim targets (the metric that matters)."""
    fis = np.where(contact[t, h])[0]
    if len(fis) == 0:
        return None
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    return float(np.sqrt(np.mean([np.sum((data.xpos[tip_ids[fi]] - tgt[t, h, fi]) ** 2)
                                  for fi in fis])))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("qpos_npz", help="Step-3 output (arm + curled fingers) to re-place")
    p.add_argument("contact_npz", help="the *_contact.npz from contact_target.py")
    p.add_argument("model_xml", help="the full model (same one Step 3 used)")
    p.add_argument("--handkp", default=None, help="(T,2,21,3) human keypoints; required for --target wrist")
    p.add_argument("--target", default="wrist", choices=("wrist", "tips"))
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model_xml)
    data = mujoco.MjData(model)
    Q = np.load(args.qpos_npz)["qpos"].copy()
    c = np.load(args.contact_npz)
    tgt, contact = c["tgt_world"], c["contact"]                          # (T,2,5,3), (T,2,5)
    T = min(len(Q), len(tgt))

    n_dof = 4 if args.target == "wrist" else 7                           # wrist mode: shoulder+elbow only
    arm = {s: arm_indices(model, s, n_dof) for s in ("left", "right")}
    tip_ids = {s: [body_id(model, b) for b in TIP_BODY[s]] for s in ("left", "right")}
    palm_id = {s: body_id(model, PALM_BODY[s]) for s in ("left", "right")}
    if args.target == "wrist":
        assert args.handkp, "--target wrist needs --handkp"
        hkp = np.load(args.handkp)                                       # (T,2,21,3); [.,.,0] = wrist

    rim_before, rim_after, place_err = [], [], []
    for t in range(T):
        for h, side in enumerate(("left", "right")):
            e0 = rim_error(model, data, Q[t], tip_ids[side], tgt, contact, t, h)
            if e0 is not None:
                rim_before.append(e0)
            if args.target == "wrist":
                e = ik_frame(model, data, Q[t], [palm_id[side]], hkp[t, h, 0][None],
                             *arm[side], args.iters)
                place_err.append(e)
            else:
                fis = np.where(contact[t, h])[0]
                if len(fis) == 0:
                    continue
                e = ik_frame(model, data, Q[t], [tip_ids[side][fi] for fi in fis],
                             tgt[t, h, fis], *arm[side], args.iters)
                place_err.append(e)
            e1 = rim_error(model, data, Q[t], tip_ids[side], tgt, contact, t, h)
            if e1 is not None:
                rim_after.append(e1)

    out = args.out or os.path.splitext(os.path.basename(args.qpos_npz))[0] + "_placed.npz"
    np.savez(out, qpos=Q)
    print(f"  placement ({args.target}) residual: {np.mean(place_err) * 100:.1f} cm")
    print(f"  fingertip->rim error (contact frames): before {np.mean(rim_before) * 100:.1f} cm"
          f"  ->  after {np.mean(rim_after) * 100:.1f} cm")
    print(f"[placed] {T} frames  ->  {out}")


if __name__ == "__main__":
    main()
