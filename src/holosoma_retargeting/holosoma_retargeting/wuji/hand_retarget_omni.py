#!/usr/bin/env python
"""Hand-only interaction-mesh retargeting for the Wuji hand (Stage B of the decoupled pipeline).

This is OmniRetarget's kinematic interaction-mesh retargeter applied HAND-ONLY: the body, wrist, and
object come from a prior body-stage solve (frozen), and only the 20 Wuji finger DOF are optimized so
the robot hand's interaction mesh (21 hand keypoints + the object's even-sampled surface points)
matches the human demonstration's, in the object frame. It reuses OmniRetarget's own object-sampling,
mesh, and Laplacian functions from src/utils.py, so the hand stage is consistent with the body stage.

Pipeline:  body/wrist/object solve (Stage A, welded fingers)  ->  THIS (fingers)  ->  render
Usage:
    python wuji/hand_retarget_omni.py <stageA_welded_npz> <out_npz> <full_wuji_model.xml> \
           <object.obj> <handkp.npy> [--pts 100]
"""
from __future__ import annotations

import os
import sys

import cvxpy as cp  # type: ignore[import-not-found]
import mujoco  # type: ignore[import-not-found]
import numpy as np
import trimesh
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # wuji/ -> repo root
sys.path.insert(0, os.path.join(REPO, "src"))
from utils import (  # noqa: E402
    calculate_laplacian_matrix,
    create_interaction_mesh,
    get_adjacency_list,
    load_object_data,
    transform_points_world_to_local,
)

# 21 robot keypoints per hand, ordered to MATCH the packer's handkp:
# palm (= wrist), then finger{1..5}_link{1..4} = MCP/PIP/DIP/tip.
HAND_BODIES = {
    "left": ["wj_left_palm_link"] + [f"wj_left_finger{i}_link{L}" for i in (1, 2, 3, 4, 5) for L in (1, 2, 3, 4)],
    "right": ["wjr_right_palm_link"] + [f"wjr_right_finger{i}_link{L}" for i in (1, 2, 3, 4, 5) for L in (1, 2, 3, 4)],
}
SIDE_PREFIX = {"left": "wj_left", "right": "wjr_right"}


class HandInteractionMeshRetargeter:
    """
    A class to perform kinematic FINGER retargeting from a human hand to the Wuji hand,
    reusing OmniRetarget's interaction mesh, with the body / wrist / object held frozen.
    """

    def __init__(self, model, side: str, object_points_local: np.ndarray, sqp_iters: int = 15, step: float = 0.3,
                 uniform: bool = True, free_wrist: bool = False, free_arm: bool = False,
                 arm_reg_weight: float = 1.0, non_pen: bool = False, obj_name: str = ""):
        """This finger retargeter solves the diffIK problem with hard constraints in SQP style,
        HAND-ONLY: the body, wrist, and object are frozen (from the body-stage solve) and only the
        20 finger DOF are optimized. During each SQP iteration, the problem is solved with the
        following constraints and costs:
            1. [Cost] Minimize the Laplacian deformation of the hand<->object interaction mesh in
               the object frame (21 hand keypoints + the even-sampled object points).
            2. [Constraint] Object non-penetration (when enabled): for every close (finger geom,
               object collision geom) pair, phi + J @ dq >= -penetration_tolerance, with phi the
               signed distance and J its relative contact Jacobian (mj_geomDistance + mj_jac).
            3. [Constraint] Enforce the finger joint limits.
            4. [Constraint] Enforce the trust region of dq.
        An infeasible QP is raised (not skipped), matching OmniRetarget's behaviour.

        Args:
            model (mujoco.MjModel): the full Wuji model (fingers free).
            side (str): which hand, "left" or "right".
            object_points_local (np.ndarray): (P, 3) object surface points in the object frame
                (OmniRetarget's even sampling), shared with the body stage.
            sqp_iters (int): number of SQP iterations per frame.
            step (float): trust-region cap on the per-iteration joint step ||dq||.
        """
        self.model = model
        self.data = mujoco.MjData(model)
        self.O = object_points_local
        self.sqp_iters = sqp_iters
        self.step = step
        self.uniform = uniform                                  # Laplacian: True=uniform (base), False=distance-weighted (1/d)

        self.kb = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in HAND_BODIES[side]]
        self.nk = len(self.kb)                                       # 21 keypoints
        pfx = SIDE_PREFIX[side]
        finger_j = [j for j in range(model.njnt)
                    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(pfx)
                    and "finger" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "")]
        # Optionally also optimize body DOF to POSITION the hand (not just curl fingers):
        #   --free-wrist -> 3 wrist DOF (orient the hand only)
        #   --free-arm   -> shoulder + elbow + wrist = 7 DOF (translate the palm to the object)
        body_names = []
        if free_arm:
            body_names = [f"{side}_shoulder_{a}_joint" for a in ("pitch", "roll", "yaw")] \
                + [f"{side}_elbow_joint"] + [f"{side}_wrist_{a}_joint" for a in ("roll", "pitch", "yaw")]
        elif free_wrist:
            body_names = [f"{side}_wrist_{a}_joint" for a in ("roll", "pitch", "yaw")]
        body_j = [j for j in (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in body_names) if j >= 0]
        fin_j = body_j + finger_j                                   # freed body DOF lead, then the fingers
        self.n_body = len(body_j)                                   # count of leading (arm/wrist) DOF
        # Regularize the freed body DOF toward Stage A ONLY for --free-arm (the arm can flail); a free
        # WRIST must roll unhindered to orient the palm, so it gets NO reg -> a fresh --free-wrist run
        # reproduces the original B0 (which had no reg).
        self.reg_idx = list(range(self.n_body)) if free_arm else []
        self.arm_reg_weight = arm_reg_weight
        self.qadr = [int(model.jnt_qposadr[j]) for j in fin_j]
        self.dadr = [int(model.jnt_dofadr[j]) for j in fin_j]
        self.lo = np.array([model.jnt_range[j][0] for j in fin_j])
        self.hi = np.array([model.jnt_range[j][1] for j in fin_j])
        self.nm = len(fin_j)                                        # 20 fingers (+3 wrist or +7 arm)

        # non-penetration (OmniRetarget constraint #2): keep finger geoms out of the object geom,
        # via mj_geomDistance (MuJoCo collides the object mesh as its CONVEX HULL -> ablation 3A).
        self.non_pen = non_pen
        if non_pen:
            # object collision geoms: the CoACD pieces if the model has them (ablation 3C), else the
            # single convex-hull mesh geom on the object body (ablation 3A).
            self.obj_geoms = [g for g in range(model.ngeom)
                              if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(f"{obj_name}_piece")]
            if not self.obj_geoms:
                ob = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{obj_name}_link")
                self.obj_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == ob]
            seen = set()
            self.finger_geoms = []                                  # one geom per finger link (20/hand)
            for g in range(model.ngeom):
                b = int(model.geom_bodyid[g])
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if nm.startswith(pfx) and "finger" in nm and b not in seen:
                    seen.add(b)
                    self.finger_geoms.append((g, b))
            # OmniRetarget's non-penetration parameters (interaction_mesh_retargeter defaults).
            self.collision_detection_threshold = 0.1                # start detecting collision within 10cm
            self.penetration_tolerance = 1e-3                       # allowed penetration when enforcing non-pen
            print(f"    [{side}] non-pen against {len(self.obj_geoms)} object collision geom(s)"
                  f" ({'CoACD pieces' if len(self.obj_geoms) > 1 else 'single convex hull'})")

    def _calc_contact_jacobian_from_point(self, body_idx, p_world):
        """
        Translational Jacobian J(q) (3 x nv) of a world point rigidly attached to a body, such that
        v_point_world = J(q) @ qvel. Analytic, via mujoco.mj_jac.

        Args:
            body_idx (int): the body the point is rigidly attached to.
            p_world (np.ndarray): the point, in world coordinates.

        Returns:
            np.ndarray: the (3, nv) translational Jacobian.
        """
        J = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, J, None, np.asarray(p_world, dtype=float), int(body_idx))
        return J

    def _compute_jacobian_for_contact_relative(self, geom1, geom2, fromto, dist):
        """
        Constraint-Jacobian row d(phi)/d(qvel) for a geom pair, where phi is the signed distance.

        The escape normal is sign(dist) * (pos1 - pos2) / ||pos1 - pos2||: when the geoms penetrate
        (dist < 0) the mj_geomDistance witness points swap sides so the raw direction flips, and the
        sign(dist) factor keeps the normal pointing outward (OmniRetarget interaction_mesh_retargeter,
        _compute_jacobian_for_contact_relative). The Jacobian is relative (geom1 body minus geom2
        body) so a moving object is handled too; here the object is frozen, so only the finger columns
        are non-zero.

        Args:
            geom1 (int): first geom id (a finger geom).
            geom2 (int): second geom id (an object collision geom).
            fromto (np.ndarray): (6,) witness segment from mj_geomDistance (pos1 on geom1, pos2 on geom2).
            dist (float): signed distance between the two geoms.

        Returns:
            np.ndarray: the (nv,) constraint-Jacobian row d(phi)/d(qvel).
        """
        pos1, pos2 = fromto[:3], fromto[3:]
        v = pos1 - pos2
        norm_v = np.linalg.norm(v)
        nhat = np.sign(dist) * (v / norm_v) if norm_v > 1e-12 else np.zeros(3)
        J_bodyA = self._calc_contact_jacobian_from_point(self.model.geom_bodyid[geom1], pos1)
        J_bodyB = self._calc_contact_jacobian_from_point(self.model.geom_bodyid[geom2], pos2)
        return nhat @ (J_bodyA - J_bodyB)

    def _update_jacobians_and_phis_from_q(self):
        """
        Non-penetration constraint data for the current pose: for every (finger geom, object collision
        geom) pair within collision_detection_threshold, the signed distance phi and its Jacobian row.

        Mirrors OmniRetarget's _update_jacobians_and_phis_from_q, restricted to the finger<->object
        pairs (this is the hand stage) and without the mj_collision broad-phase prefilter -- our pair
        set is tiny (finger geoms x object pieces), so the exhaustive mj_geomDistance sweep is already
        cheap. Assumes mujoco.mj_forward has been called for the current qpos (done by the caller).

        Returns:
            tuple: (Js, phis) - dicts keyed by (finger_geom, object_geom); Js[key] is the (nv,)
                Jacobian row and phis[key] is the signed distance.
        """
        m, d = self.model, self.data
        threshold = self.collision_detection_threshold
        Js, phis = {}, {}
        fromto = np.zeros(6)
        for g1, _b in self.finger_geoms:
            for g2 in self.obj_geoms:
                fromto[:] = 0.0
                dist = mujoco.mj_geomDistance(m, d, g1, g2, threshold, fromto)
                if dist < threshold:
                    Js[(g1, g2)] = self._compute_jacobian_for_contact_relative(g1, g2, fromto, dist)
                    phis[(g1, g2)] = float(dist)
        return Js, phis

    def solve_single_iteration(self, qfull, q_move, Lk, lap_src, lapO, quat, trans, frame_idx, q_ref):
        """Solve a single SQP iteration of the finger diffIK problem.

        Linearizes the interaction-mesh Laplacian via the MuJoCo analytic Jacobian and solves a QP
        for the finger step dq under joint limits and a trust region (Clarabel).

        Args:
            qfull (np.ndarray): full qpos with the frozen body/wrist/object for this frame.
            q_move (np.ndarray): current finger angles (nm,).
            Lk (np.ndarray): keypoint block of the Laplacian, (n, nk).
            lap_src (np.ndarray): target Laplacian coordinates, (n, 3).
            lapO (np.ndarray): fixed object contribution to the Laplacian, (n, 3).
            quat, trans (np.ndarray): object pose (wxyz, xyz) for the object-local frame.
            frame_idx (int): current frame index (for the infeasibility error message).
            q_ref (np.ndarray): Stage-A pose of the optimized DOF (nm,); the regularizer target for
                the freed body (arm/wrist) DOF when --free-arm/--free-wrist is on.

        Returns:
            np.ndarray: the joint step dq (nm,).

        Raises:
            RuntimeError: if the QP is infeasible even after relaxing the trust region (faithful to
                OmniRetarget: an infeasible non-penetration QP stops the solve).
        """
        qfull[self.qadr] = q_move
        self.data.qpos[:] = qfull
        mujoco.mj_forward(self.model, self.data)
        Rt = trimesh.transformations.quaternion_matrix(quat)[:3, :3]           # object world rotation
        pk_world = np.array([self.data.xpos[b] for b in self.kb])
        pk_loc = transform_points_world_to_local(quat, trans, pk_world)        # robot keypoints -> object-local
        J = np.zeros((self.nk, 3, self.nm))
        for i, b in enumerate(self.kb):
            jp = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jp, None, self.data.xpos[b], b)
            J[i] = Rt.T @ jp[:, self.dadr]                                      # object-local Jacobian (3, nm)
        dq = cp.Variable(self.nm)
        dpk = cp.vstack([J[i] @ dq for i in range(self.nk)])
        lap = Lk @ (pk_loc + dpk) + lapO                                       # linearized Laplacian
        # Constraints: finger joint limits + object non-penetration; trust region kept separate so it
        # can be relaxed on a solve failure (OmniRetarget's fallback).
        cons = [q_move + dq >= self.lo, q_move + dq <= self.hi]
        if self.non_pen:
            Js, phis = self._update_jacobians_and_phis_from_q()
            for key, phi in phis.items():                                       # enforce phi + J @ dq >= -tolerance
                cons.append(phi + Js[key][self.dadr] @ dq >= -self.penetration_tolerance)
        soc = cp.norm(dq) <= self.step                                          # trust region

        # Solve with Clarabel; if it fails, drop the trust region and retry; if it STILL fails, raise
        # -- faithful to OmniRetarget: an infeasible non-penetration QP stops the solve, it is NOT
        # silently skipped (interaction_mesh_retargeter.solve_single_iteration).
        obj_terms = [cp.sum_squares(lap - lap_src)]
        if self.reg_idx and self.arm_reg_weight > 0:                            # keep freed body DOF near Stage A
            ri = self.reg_idx
            obj_terms.append(self.arm_reg_weight * cp.sum_squares((q_move[ri] + dq[ri]) - q_ref[ri]))
        objective = cp.Minimize(cp.sum(obj_terms))
        problem = cp.Problem(objective, cons + [soc])
        problem.solve(solver=cp.CLARABEL)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            problem = cp.Problem(objective, cons)
            problem.solve(solver=cp.CLARABEL)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"CVXPY solve failed: {problem.status} (frame {frame_idx})")
        return dq.value

    def retarget_fingers(self, Q, hkp, obj_quat, obj_pos):
        """Solve the finger DOF for a whole trajectory, warm-started per frame.

        Per frame: express the human hand keypoints in the object frame, (re)build the Delaunay
        interaction mesh + uniform Laplacian over [hand keypoints + object points], then iterate
        solve_single_iteration. The solved finger angles are written back into Q in place.

        Args:
            Q (np.ndarray): (T, nq) full qpos; body/wrist/object frozen, finger columns overwritten.
            hkp (np.ndarray): (T, 21, 3) human hand keypoints in world frame.
            obj_quat (np.ndarray): (T, 4) object orientation per frame (wxyz).
            obj_pos (np.ndarray): (T, 3) object translation per frame.
        """
        T = Q.shape[0]
        n = self.nk + self.O.shape[0]
        q_move = Q[0, self.qadr].copy()                             # warm start (fingers open)
        out = np.zeros((T, self.nm))
        for t in tqdm(range(T), desc="  fingers", leave=False):
            hloc = transform_points_world_to_local(obj_quat[t], obj_pos[t], hkp[t])   # human kp -> object-local
            src = np.vstack([hloc, self.O])                                           # interaction-mesh vertices
            _, tets = create_interaction_mesh(src)                                    # Delaunay (OmniRetarget)
            adj = get_adjacency_list(tets, n)
            L = calculate_laplacian_matrix(src, adj, uniform_weight=self.uniform)     # uniform (base) or distance-weighted
            lap_src = L @ src
            Lk, lapO = L[:, :self.nk], L[:, self.nk:] @ self.O                        # split fixed object part
            qfull = Q[t].copy()
            q_ref = Q[t, self.qadr].copy()                         # Stage-A pose of the optimized DOF (regularizer target)
            q_move[self.reg_idx] = q_ref[self.reg_idx]             # start the freed body DOF from Stage A each frame
            for _ in range(self.sqp_iters):
                dq = self.solve_single_iteration(qfull, q_move, Lk, lap_src, lapO, obj_quat[t], obj_pos[t], t, q_ref)
                q_move = q_move + dq
            out[t] = q_move
        for k, a in enumerate(self.qadr):
            Q[:, a] = out[:, k]


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    stageA_npz, out_npz, model_xml, object_obj, handkp_npy = args[:5]
    n_obj = int(sys.argv[sys.argv.index("--pts") + 1]) if "--pts" in sys.argv else 100
    uniform = "--distance-weighted" not in sys.argv                  # ablation: Laplacian weighting knob
    free_wrist = "--free-wrist" in sys.argv                          # ablation: also optimize the 3 wrist DOF
    free_arm = "--free-arm" in sys.argv                              # ablation: also optimize shoulder+elbow+wrist (7 DOF)
    non_pen = "--non-pen" in sys.argv                                # ablation 3A: object non-penetration (convex hull)
    obj_name = os.path.splitext(os.path.basename(object_obj))[0]     # object geom name (e.g. "table")

    Mf = mujoco.MjModel.from_xml_path(model_xml)                     # full model (fingers free, + connector)
    # Stage-A welded model: fingers welded, no object collision pieces (CoACD only matters for the
    # finger non-pen here), so strip any _coacd suffix from the full-model path.
    Mb = mujoco.MjModel.from_xml_path(model_xml.replace("_wuji_w_", "_wuji_welded_w_").replace("_coacd", ""))
    npz = np.load(stageA_npz, allow_pickle=True)
    qb = npz["qpos"]
    T = qb.shape[0]
    hkp = np.load(handkp_npy)                                        # (T, 2, 21, 3): [left, right]

    # full-DOF qpos: base/object/body/arm from the Stage-A welded solve, fingers 0 to start
    Q = np.zeros((T, Mf.nq))
    Q[:, 0:7] = qb[:, 0:7]
    Q[:, Mf.nq - 7:] = qb[:, Mb.nq - 7:]
    for j in range(Mf.njnt):
        nm = mujoco.mj_id2name(Mf, mujoco.mjtObj.mjOBJ_JOINT, j)
        if nm and "finger" not in nm:
            jb = mujoco.mj_name2id(Mb, mujoco.mjtObj.mjOBJ_JOINT, nm)
            if jb >= 0:
                Q[:, int(Mf.jnt_qposadr[j])] = qb[:, int(Mb.jnt_qposadr[jb])]

    obj_pos = Q[:, Mf.nq - 7:Mf.nq - 4]
    obj_quat = Q[:, Mf.nq - 4:Mf.nq]                                 # object freejoint (wxyz)
    O, _ = load_object_data(object_obj, smpl_scale=1.0, sample_count=n_obj)   # even sampling, object-local, metres
    dof_desc = "arm+fingers" if free_arm else ("wrist+fingers" if free_wrist else "fingers")
    print(f"[hand-omni] {T} frames | {O.shape[0]} object points (even, requested {n_obj}) | 21 keypoints/hand"
          f" | Laplacian={'uniform (base)' if uniform else 'distance-weighted'} | DOF={dof_desc}"
          f" | non-pen={'ON' if non_pen else 'off'}")
    for si, side in enumerate(("left", "right")):
        print(f"  solving {side} fingers ...")
        HandInteractionMeshRetargeter(Mf, side, O, uniform=uniform, free_wrist=free_wrist, free_arm=free_arm,
                                      non_pen=non_pen, obj_name=obj_name).retarget_fingers(
            Q, hkp[:, si], obj_quat, obj_pos)

    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    extra = {"human_joints": npz["human_joints"]} if "human_joints" in npz.files else {}
    np.savez(out_npz, qpos=Q, fps=30, **extra)
    print(f"  wrote {out_npz}  qpos {Q.shape}")


if __name__ == "__main__":
    main()
