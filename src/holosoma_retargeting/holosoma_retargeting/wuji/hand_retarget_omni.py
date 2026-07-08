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

    def __init__(self, model, side: str, object_points_local: np.ndarray, sqp_iters: int = 15, step: float = 0.3):
        """This finger retargeter solves the diffIK problem with hard constraints in SQP style,
        HAND-ONLY: the body, wrist, and object are frozen (from the body-stage solve) and only the
        20 finger DOF are optimized. During each SQP iteration, the problem is solved with the
        following constraints and costs:
            1. [Cost] Minimize the Laplacian deformation of the hand<->object interaction mesh in
               the object frame (21 hand keypoints + the even-sampled object points).
            2. [Constraint] Enforce the finger joint limits.
            3. [Constraint] Enforce the trust region of dq.

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

        self.kb = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in HAND_BODIES[side]]
        self.nk = len(self.kb)                                       # 21 keypoints
        pfx = SIDE_PREFIX[side]
        fin_j = [j for j in range(model.njnt)
                 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(pfx)
                 and "finger" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "")]
        self.qadr = [int(model.jnt_qposadr[j]) for j in fin_j]
        self.dadr = [int(model.jnt_dofadr[j]) for j in fin_j]
        self.lo = np.array([model.jnt_range[j][0] for j in fin_j])
        self.hi = np.array([model.jnt_range[j][1] for j in fin_j])
        self.nm = len(fin_j)                                        # 20 finger DOF

    def solve_single_iteration(self, qfull, q_move, Lk, lap_src, lapO, quat, trans):
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

        Returns:
            np.ndarray | None: the joint step dq (nm,), or None if the QP failed.
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
        cons = [q_move + dq >= self.lo, q_move + dq <= self.hi, cp.norm(dq) <= self.step]
        try:
            cp.Problem(cp.Minimize(cp.sum_squares(lap - lap_src)), cons).solve(solver=cp.CLARABEL)
        except Exception:
            return None
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
            L = calculate_laplacian_matrix(src, adj, uniform_weight=True)             # OmniRetarget uniform Laplacian
            lap_src = L @ src
            Lk, lapO = L[:, :self.nk], L[:, self.nk:] @ self.O                        # split fixed object part
            qfull = Q[t].copy()
            for _ in range(self.sqp_iters):
                dq = self.solve_single_iteration(qfull, q_move, Lk, lap_src, lapO, obj_quat[t], obj_pos[t])
                if dq is None:
                    break
                q_move = q_move + dq
            out[t] = q_move
        for k, a in enumerate(self.qadr):
            Q[:, a] = out[:, k]


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    stageA_npz, out_npz, model_xml, object_obj, handkp_npy = args[:5]
    n_obj = int(sys.argv[sys.argv.index("--pts") + 1]) if "--pts" in sys.argv else 100

    Mf = mujoco.MjModel.from_xml_path(model_xml)                     # full model (fingers free, + connector)
    Mb = mujoco.MjModel.from_xml_path(model_xml.replace("_wuji_w_", "_wuji_welded_w_"))   # Stage-A welded model
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
    print(f"[hand-omni] {T} frames | {O.shape[0]} object points (even, requested {n_obj}) | 21 keypoints/hand")
    for si, side in enumerate(("left", "right")):
        print(f"  solving {side} fingers ...")
        HandInteractionMeshRetargeter(Mf, side, O).retarget_fingers(Q, hkp[:, si], obj_quat, obj_pos)

    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    extra = {"human_joints": npz["human_joints"]} if "human_joints" in npz.files else {}
    np.savez(out_npz, qpos=Q, fps=30, **extra)
    print(f"  wrote {out_npz}  qpos {Q.shape}")


if __name__ == "__main__":
    main()
