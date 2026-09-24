#!/usr/bin/env python
"""TopoRetarget on a FLOATING hand + object — no body, no arm, no Stage A placement (hand-only track).

The full pipeline entangles the hand-object solve with Stage A's body fit and the G1 arm's reach; the
placement experiments showed those dominate the failure. This driver removes them: the hand hangs from
a 6-DOF root (make_hand_model.py) and is INITIALIZED WHERE THE HUMAN HAND IS -- per frame, the rigid
palm points (wrist + 5 MCPs) of the robot's rest hand are Kabsch-fitted onto the human's, giving the
root pose that best overlays the robot hand on the human hand. The solver (TopoRetargeter, unchanged)
then frees 6 root + 20 finger DOF, so the only thing under test is the hand-object objective itself:
initialized AT the grasp, can it hold and articulate one?

The first positional arg supplies ONLY the object trajectory, from either source:
  * a Stage-A `*_original.npz` (tail of its qpos) -- the robot-frame trajectory, comparable to the
    full-body runs; or
  * a `humoto_pt/<seq>.pt` (humoto_direct_to_pt.py) -- pos at [:,318:321], quat XYZW at [:,321:325],
    which is the SAME z-up floor-normalized frame the handkp is written in, BY CONSTRUCTION (one
    script writes both). This is the path for sequences that never went through Stage A.

Init sanity is printed at frame 0 ("init palm residual"): the Kabsch pose is pushed through the actual
kinematics and compared back to the fit, so an Euler-order or offset mistake shows up as centimetres,
not as a silently wrong experiment.

Usage (omniretarget env), run from the retargeting root:
    python wuji/hand_table_topo.py \
        demo_results/.../<seq>_original.npz out.npz models/g1/hand_right_w_table_coacd.xml \
        models/table/table.obj <handkp.npy> \
        [--side right] [--bone] [--exp-weight] [--smooth] [--warmup] [--soft-pen] [--pts N]
"""
from __future__ import annotations

import os
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topoRetarget import HAND_BODIES, TopoRetargeter, human_kp_indices  # noqa: E402
from utils import load_object_data  # noqa: E402

PALM_RIGID = [0, 1, 5, 9, 13, 17]        # wrist + the 5 MCPs: the (near-)rigid palm subset of the 21


def kabsch(src: np.ndarray, dst: np.ndarray):
    """Rigid transform (R, t) minimizing ||R @ src + t - dst||, reflection-guarded.

    Args:
        src: (N, 3) source points.
        dst: (N, 3) target points.

    Returns:
        tuple: (R (3,3), t (3,)).
    """
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    R = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
    return R, cd - R @ cs


def rest_keypoints(model, side: str):
    """The robot hand's 21 keypoint positions at zero qpos, plus the root body's default pose.

    Returns:
        tuple: (kp (21,3) world at rest, root_pos0 (3,)).
    """
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    kb = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in HAND_BODIES[side]]
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"hand_root_{side}")   # two-hand model
    if root < 0:
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_root")       # single-hand model
    return np.array([d.xpos[b] for b in kb]), d.xpos[root].copy()


def scale_hand_to_robot(hkp: np.ndarray, robot_rest: np.ndarray) -> np.ndarray:
    """Rescale the HUMAN hand keypoints, bone by bone, to the ROBOT's segment lengths.

    The Wuji fingers are LONGER than the human's, and nothing in the interaction-mesh objective knows
    it (the Laplacian absorbs the mismatch as residual; the contact targets end up spaced for a human
    hand). Measured consequence: every fingertip hyperextends 11-16 deg trying to reach targets that
    sit too close to the palm. The fix, done at the SOURCE like the Wuji repo's own per-segment
    scaling: walk each finger chain (wrist -> MCP -> PIP -> DIP -> tip) and rescale every segment
    vector by (robot length / human median length), so the "human" the solver imitates has the
    robot's proportions. Directions -- the pose -- are untouched; only lengths change, so the bone
    (angle) term sees the same hand and everything downstream (Kabsch init, contact targets,
    Laplacian source) becomes robot-consistent.

    Args:
        hkp: (T, 21, 3) human keypoints, MediaPipe-style layout (0 = wrist, then 5x MCP/PIP/DIP/tip).
        robot_rest: (21, 3) the robot hand's keypoints at rest (rest_keypoints).

    Returns:
        np.ndarray: (T, 21, 3) the rescaled keypoints.
    """
    # NB: scale per WHOLE FINGER, not per segment. The robot's intermediate link origins do NOT
    # correspond one-to-one to human PIP/DIP -- Wuji stacks two joints at the knuckle, so its
    # link1->link2 "segment" is near zero-length and per-segment ratios come out absurd (0.16, 2.28),
    # deforming the hand. Only the ENDPOINTS are semantically solid (knuckle, tip), so each finger
    # gets ONE ratio (robot chain length / human chain length) applied to all its segments -- total
    # reach matches the robot, the human's internal finger proportions are preserved. The palm
    # segment (wrist->knuckle) gets its own ratio; knuckle positions are meaningful on both.
    chains = [(0, 1 + 4 * f, 2 + 4 * f, 3 + 4 * f, 4 + 4 * f) for f in range(5)]
    seg_len = lambda kp, a, b: np.linalg.norm(kp[..., b, :] - kp[..., a, :], axis=-1)
    out = hkp.copy()
    print("  hand scaling (robot/human): palm-seg ratio + one whole-finger ratio each:")
    for f, ch in enumerate(chains):
        r_palm = float(seg_len(robot_rest[None], ch[0], ch[1])[0]
                       / max(np.median(seg_len(hkp, ch[0], ch[1])), 1e-6))
        robot_chain = sum(float(seg_len(robot_rest[None], a, b)[0])
                          for a, b in zip(ch[1:-1], ch[2:]))
        human_chain = sum(float(np.median(seg_len(hkp, a, b)))
                          for a, b in zip(ch[1:-1], ch[2:]))
        r_fing = robot_chain / max(human_chain, 1e-6)
        out[:, ch[1]] = out[:, ch[0]] + r_palm * (hkp[:, ch[1]] - hkp[:, ch[0]])
        for a, b in zip(ch[1:-1], ch[2:]):
            out[:, b] = out[:, a] + r_fing * (hkp[:, b] - hkp[:, a])
        print(f"    finger{f + 1}: palm {r_palm:.2f}   finger {r_fing:.2f}")
    return out


def smooth_contact_mask(mask: np.ndarray, enter: int = 3, exit_: int = 5, ramp: int = 5) -> np.ndarray:
    """Turn the raw per-frame contact mask into a debounced, RAMPED 0..1 weight.

    The raw mask flickers as fingertips cross the tau threshold, and each flip adds/removes a
    weight-50 objective term between frames -- measured as the cause of 12 of the 22 jitter spikes.
    Hysteresis (enter after `enter` consecutive contact frames, leave after `exit_` clear ones) kills
    the flicker; the moving-average ramp fades the term in/out over ~`ramp` frames instead of stepping
    it, so the optimum moves continuously.

    Args:
        mask: (T, 5) bool, raw human-contact mask.
        enter, exit_: debounce frame counts.
        ramp: fade width in frames.

    Returns:
        np.ndarray: (T, 5) float weights in [0, 1].
    """
    T = len(mask)
    w = np.zeros((T, mask.shape[1]), np.float64)
    for fi in range(mask.shape[1]):
        state, cin, cout = False, 0, 0
        st = np.zeros(T, bool)
        for t in range(T):
            cin, cout = (cin + 1, 0) if mask[t, fi] else (0, cout + 1)
            if not state and cin >= enter:
                state = True
            if state and cout >= exit_:
                state = False
            st[t] = state
        w[:, fi] = np.convolve(st.astype(float), np.ones(ramp) / ramp, mode="same")
    return w


def smooth_targets(tgt_loc: np.ndarray, mesh, win: int = 9) -> np.ndarray:
    """Temporally smooth the contact targets, then put them back ON the object surface.

    The raw target is the surface point nearest the HUMAN fingertip, recomputed independently every
    frame: mocap wiggle moves it a few mm, and near the table EDGE the nearest-point can flip between
    the top face and the rim face -- centimetre jumps that a weight-50 attraction faithfully tracks
    (the residual shake seen while the hand is pressed on the table). A moving average kills both; the
    re-projection keeps the averaged point on the mesh (averaging across an edge cuts the corner).

    Args:
        tgt_loc: (T, 5, 3) object-local contact targets.
        mesh: the object trimesh (local frame).
        win: smoothing window, frames.

    Returns:
        np.ndarray: (T, 5, 3) smoothed, surface-projected targets.
    """
    import trimesh
    from scipy.ndimage import uniform_filter1d

    sm = uniform_filter1d(tgt_loc.astype(np.float64), size=win, axis=0, mode="nearest")
    pq = trimesh.proximity.ProximityQuery(mesh)
    T = len(sm)
    flat, _, _ = pq.on_surface(sm.reshape(T * 5, 3))
    return flat.reshape(T, 5, 3).astype(np.float32)


def init_root_from_human(model, retargeter, Q: np.ndarray, hkp: np.ndarray, side: str):
    """Set every frame's 6 root DOF so the robot's rest palm overlays the HUMAN's palm.

    Per frame: Kabsch-fit the robot's rest palm-rigid points onto the human's, then convert the
    fitted rigid transform into the root's slide values + intrinsic-XYZ hinge angles. The result is
    verified through the real kinematics at frame 0 (a wrong Euler convention would silently rotate
    the whole experiment; the printed residual makes that impossible to miss).

    Args:
        model: the floating-hand model.
        retargeter: the built TopoRetargeter (for qadr of the 6 root DOF -- the first 6 of qadr).
        Q: (T, nq) qpos to fill, in place.
        hkp: (T, 21, 3) human hand keypoints, world.
        side: "left"/"right".
    """
    rest, root_pos0 = rest_keypoints(model, side)
    manifold = getattr(retargeter, "base_manifold", False)
    if manifold:                                                    # S3 base: 3 slides + a ball quaternion
        ta = retargeter._trans_qadr; ba = retargeter._ball_qadr
        for t in range(len(Q)):
            R, tr = kabsch(rest[PALM_RIGID], hkp[t][PALM_RIGID])
            Q[t, ta] = (R @ root_pos0 + tr) - root_pos0             # slides act in the world frame (same as Euler)
            xyzw = Rot.from_matrix(R).as_quat()
            Q[t, ba:ba + 4] = xyzw[[3, 0, 1, 2]]                     # ball qpos = wxyz (MuJoCo order)
    else:
        root_adr = retargeter.qadr[:6]                               # x, y, z, roll, pitch, yaw
        for t in range(len(Q)):
            R, tr = kabsch(rest[PALM_RIGID], hkp[t][PALM_RIGID])
            Q[t, root_adr[:3]] = (R @ root_pos0 + tr) - root_pos0    # slides act in the world frame
            Q[t, root_adr[3:]] = Rot.from_matrix(R).as_euler("XYZ")  # stacked hinges = intrinsic XYZ

    # push frame 0 through the real kinematics and measure how far the palm landed from the fit
    d = mujoco.MjData(model)
    d.qpos[:] = Q[0]
    mujoco.mj_forward(model, d)
    kb = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in HAND_BODIES[side]]
    got = np.array([d.xpos[kb[i]] for i in PALM_RIGID])
    R, tr = kabsch(rest[PALM_RIGID], hkp[0][PALM_RIGID])
    want = (R @ rest[PALM_RIGID].T).T + tr
    print(f"  init palm residual (frame 0): {np.linalg.norm(got - want, axis=1).mean() * 100:.2f} cm"
          f"   [~0 = Euler/offset conventions correct]")


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    stageA_npz, out_npz, model_xml, object_obj, handkp_npy = args[:5]
    side = sys.argv[sys.argv.index("--side") + 1] if "--side" in sys.argv else "right"
    n_obj = int(sys.argv[sys.argv.index("--pts") + 1]) if "--pts" in sys.argv else 100
    # --demo-scale S: uniformly scale the Stage-B DEMO (hand keypoints + object translation + object
    # interaction points) by S, orientations UNCHANGED -- exactly what OmniRetarget does to the human
    # joints + object in Stage A. Use S = robot_height/human_height (0.819 for HUMOTO) so the floating-hand
    # grasp is solved at the SAME coordinate scale as the Stage-A body (closes the ~14cm height offset).
    # scale_hand_to_robot (--scale-hand) still handles the finger-shape mismatch separately.
    demo_scale = float(sys.argv[sys.argv.index("--demo-scale") + 1]) if "--demo-scale" in sys.argv else 1.0
    obj_name = os.path.splitext(os.path.basename(object_obj))[0]
    finger_kp = ("mcp_tip" if "--drop-pip-dip" in sys.argv else            # [MCP,TIP] only (no E_bone)
                 "remap" if "--remap-fingers" in sys.argv else "full")     # remap = [MCP,PIP,DIP,TIP]

    model = mujoco.MjModel.from_xml_path(model_xml)
    if stageA_npz.endswith(".pt"):
        # humoto_pt packer output: object pose in the handkp's own frame. Quat is stored XYZW
        # (scipy as_quat); MuJoCo's freejoint wants WXYZ, hence the [3,0,1,2] reorder.
        import torch
        pt = torch.load(stageA_npz).numpy()
        qb = np.concatenate([pt[:, 318:321], pt[:, 321:325][:, [3, 0, 1, 2]]], axis=1)
    else:
        qb = np.load(stageA_npz, allow_pickle=True)["qpos"]
    hk = np.load(handkp_npy)                                         # (T, 2, 21, 3): [left, right]
    if demo_scale != 1.0:                                            # OmniRetarget-match: uniform-scale the DEMO
        hk = hk * demo_scale                                         #   hand keypoints (orientation unchanged)
        qb = np.array(qb, float); qb[:, -7:-4] = qb[:, -7:-4] * demo_scale   # object translation xyz (quat unchanged)
        print(f"  demo-scale {demo_scale}: handkp + object translation + object points scaled to OmniRetarget frame")
    hkp = hk[:, 0 if side == "left" else 1]
    T = min(len(qb), len(hkp))

    if "--scale-hand" in sys.argv:
        rest, _ = rest_keypoints(model, side)
        hkp = scale_hand_to_robot(hkp, rest)

    # --paper-faithful: "TopoRetarget paper-parameter adaptation with Kabsch-referenced floating-base
    # regularization and wrist-frame bone directions" (arXiv:2606.16272) -- NOT a bit-exact reproduction.
    # Faithful: E_IM=500/N_v, exp Laplacian k=30, wrist-frame direction-difference E_bone, weights (bone 0.1 /
    # warmup 1.0,2.5 / smooth 2.5), soft-pen params + FULL hand collision coverage, extras off, anatomical
    # remap, EXACTLY 50 object pts, warmup Eq.2 (bone+smooth only, no base prior).
    # DOCUMENTED ADAPTATIONS (paper under-specifies): base-pose prior applies the paper's 100/1 weights to
    # deviation from the per-frame KABSCH base (paper writes ||q_base||^2 vs an unstated reference); warmup is
    # a fixed 8 iters; SQP stops at ||dq||<1e-4 (max 30); joint limits + trust region are numerical safeguards.
    paper = "--paper-faithful" in sys.argv
    if paper:
        finger_kp, n_obj = "remap", 50
    # exact-count area-weighted sampling for paper mode (Poisson-even CAPS below the requested count)
    O, _ = load_object_data(object_obj, smpl_scale=1.0, sample_count=n_obj,
                            surface_weights=(lambda p: 1.0) if paper else None)
    if paper:
        assert O.shape[0] == 50, f"paper mode needs exactly 50 object points, got {O.shape[0]}"
    kw = dict(
        exp_weight="--exp-weight" in sys.argv, use_bone="--bone" in sys.argv,
        use_smooth="--smooth" in sys.argv, use_warmup="--warmup" in sys.argv,
        soft_pen="--soft-pen" in sys.argv, obj_name=obj_name,
        kappa=float(sys.argv[sys.argv.index("--kappa") + 1]) if "--kappa" in sys.argv else 30.0,
        w_bone=float(sys.argv[sys.argv.index("--w-bone") + 1]) if "--w-bone" in sys.argv else 0.1,
        w_smooth=float(sys.argv[sys.argv.index("--w-smooth") + 1]) if "--w-smooth" in sys.argv else 0.05,
        # floating-base prior: pull the 6 root DOF toward the per-frame Kabsch (human-hand) pose. Keep it SMALL.
        wrist_reg=float(sys.argv[sys.argv.index("--wrist-reg") + 1]) if "--wrist-reg" in sys.argv else 0.0,
        w_contact=(float(sys.argv[sys.argv.index("--w-contact") + 1]) if "--w-contact" in sys.argv
                   else 50.0) if "--contact" in sys.argv else 0.0,
        vel_bound="--vel-bound" in sys.argv,
        vmax_trans=float(sys.argv[sys.argv.index("--vmax-trans") + 1]) if "--vmax-trans" in sys.argv else 0.03,
        vmax_rot=float(sys.argv[sys.argv.index("--vmax-rot") + 1]) if "--vmax-rot" in sys.argv else 0.15,
        self_col="--self-col" in sys.argv,
        tip_mode=sys.argv[sys.argv.index("--tip-kp") + 1] if "--tip-kp" in sys.argv else "joint",
        w_tip=float(sys.argv[sys.argv.index("--w-tip") + 1]) if "--w-tip" in sys.argv else 1.0,
        w_dir=float(sys.argv[sys.argv.index("--w-dir") + 1]) if "--w-dir" in sys.argv
              else (0.1 if "--dir-term" in sys.argv else 0.0),
        finger_kp=finger_kp, surface_pq=None)
    if paper:                                                          # TopoRetarget paper config (see note above)
        wi = int(sys.argv[sys.argv.index("--warmup-iters") + 1]) if "--warmup-iters" in sys.argv else 8  # ablation knob
        kw.update(exp_weight=True, use_bone=True, bone_mode="wrist", w_bone=0.1,   # wrist-frame dir-diff E_bone
                  use_warmup=(wi > 0), warmup_iters=wi, warmup_w_bone=1.0, warmup_w_smooth=2.5,
                  use_smooth=True, w_smooth=2.5,
                  base_reg_trans=100.0, base_reg_rot=1.0, wrist_reg=0.0,   # base reg vs previous frame (trans/rot split)
                  lambda_im=500.0, im_normalize=True,                      # E_IM: 500 / N_v
                  soft_pen=True, sqp_tol=1e-4, sqp_iters=30, finger_kp="remap",
                  w_contact=0.0, vel_bound=False, self_col=False, w_tip=1.0, w_dir=0.0)  # extras OFF
    ret = TopoRetargeter(model, side, O, **kw)

    # qpos: object trajectory at the tail (from Stage A, same as every other run); root from the HUMAN
    Q = np.zeros((T, model.nq))
    Q[:, model.nq - 7:] = qb[:T, -7:]
    init_root_from_human(model, ret, Q, hkp[:T], side)

    # clamp the init into the joint limits: an out-of-range start makes the FIRST QP infeasible and
    # the solver returns None forever (= a hand frozen at frame 0). Not hypothetical -- the Wuji
    # thumb's joint1 range is [+0.048, 1.603], so "fingers at 0" is already illegal.
    q0 = np.clip(Q[:, ret.qadr], ret.lo, ret.hi)
    n_clamped = int((q0 != Q[:, ret.qadr]).any(axis=0).sum())
    Q[:, ret.qadr] = q0
    if n_clamped:
        print(f"  init clamp: {n_clamped} DOF nudged inside their joint limits")

    obj_pos = Q[:, model.nq - 7:model.nq - 4]
    obj_quat = Q[:, model.nq - 4:]

    # contact attraction (--contact): where does the HUMAN touch the object? The targets carry the
    # contact TOPOLOGY the Laplacian cannot see -- e.g. the thumb's target is on the RIM FACE while the
    # other fingers' targets are on the TOP; this is the only term that knows the difference.
    anchor_arg = sys.argv[sys.argv.index("--anchor") + 1] if "--anchor" in sys.argv else "0"
    contact_tgt = contact_mask = None
    if ret.w_contact > 0 or anchor_arg == "auto":
        import trimesh
        from contact_target import fingertip_contacts
        # NB uses the (possibly robot-rescaled) hkp: the targets are then spaced for the ROBOT's reach
        h2 = np.stack([hkp[:T], hkp[:T]], axis=1)
        tgt_w, tgt_l, cmask, dist = fingertip_contacts(
            h2, trimesh.load(object_obj, force="mesh"), obj_pos, obj_quat, tau=0.03)
        if ret.w_contact > 0:
            import trimesh as _tm
            # debounce+ramp the mask AND temporally smooth the targets themselves -- both halves of
            # the shake (term popping in/out; targets wiggling/face-flipping at the table edge).
            # --raw-tgt reproduces the unsmoothed behaviour, for A/B ablations only.
            if "--raw-tgt" in sys.argv:
                contact_tgt = tgt_l[:, 0]
            else:
                contact_tgt = smooth_targets(tgt_l[:, 0], _tm.load(object_obj, force="mesh"))
            contact_mask = smooth_contact_mask(cmask[:, 0])
            print(f"  contact attraction: w={ret.w_contact}, human in contact "
                  f"{int((contact_mask.max(axis=1) > 0).sum())}/{T} frames "
                  f"({'RAW targets' if '--raw-tgt' in sys.argv else 'debounced+ramped+smoothed'})")
        if "--reach-tgt" in sys.argv:
            import trimesh as _tm
            ret.surface_pq = _tm.proximity.ProximityQuery(_tm.load(object_obj, force="mesh"))
            print(f"  reachable targets: nearest-surface within {ret.reach_r * 100:.0f} cm of the human contact")

    # --anchor auto: start the solve at the frame with the MOST fingers in contact (the grasp) and
    # propagate outward, so the most-constrained moment defines the basin, not the approach.
    anchor = int(np.argmax(cmask[:, 0].sum(axis=1))) if anchor_arg == "auto" else int(anchor_arg)
    if anchor:
        print(f"  anchor frame: {anchor} (solve runs {anchor}->end, then {anchor - 1}->0)")

    # subset the human keypoints to match the robot keypoint set (self.kb). init/contact above keep the
    # full 21 (they use PALM_RIGID / fingertip_contacts); only the interaction mesh uses the reduced set.
    hkp_mesh = hkp[:, human_kp_indices(finger_kp)]
    print(f"[hand-topo] {T} frames | {O.shape[0]} object pts | side={side} | finger_kp={finger_kp} "
          f"({ret.nk} keypoints) | DOF={ret.nm} | soft-pen={'ON' if ret.soft_pen else 'off'}")
    ret.retarget_fingers(Q, hkp_mesh[:T], obj_quat, obj_pos,
                         contact_tgt=contact_tgt, contact_mask=contact_mask, anchor=anchor)

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez(out_npz, qpos=Q)
    print(f"[hand-topo] -> {out_npz}")


if __name__ == "__main__":
    main()
