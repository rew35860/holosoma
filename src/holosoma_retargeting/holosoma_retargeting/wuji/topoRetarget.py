#!/usr/bin/env python
"""TopoRetarget hand retargeting (Wu et al. 2026, arXiv:2606.16272) for the Wuji hand — Stage B.

OmniRetarget's interaction mesh (wuji/hand_retarget_omni.py) plus three ablatable upgrades:
    --exp-weight  exp distance-weighted Laplacian w_ij = softmax exp(-kappa*d) (kappa=30, Eq.5) vs uniform.
    --bone        E_bone (Eq.1): match adjacent finger-bone directions -> preserves the curl the Laplacian misses.
    --soft-pen    slack object non-penetration (Eq.8): phi+s >= -tau soft, phi >= -b hard; never infeasible.

Else identical to hand_retarget_omni (clean A/B): 21 keypoints, shared Delaunay graph, per-frame
warm-started SQP (MuJoCo Jacobian + Clarabel), free-wrist base. Graph/weights built once on the human.

Pipeline:  Stage A (welded body/wrist/object)  ->  THIS (wrist + fingers)  ->  render
Usage:
    python wuji/topoRetarget.py <stageA_welded_npz> <out_npz> <full_wuji_model.xml> \
           <object.obj> <handkp.npy> [--bone] [--exp-weight] [--soft-pen] [--pts N] [--kappa 30]
"""
from __future__ import annotations

import os
import sys

import cvxpy as cp  # type: ignore[import-not-found]
import mujoco  # type: ignore[import-not-found]
import numpy as np
import trimesh
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from utils import (  # noqa: E402
    create_interaction_mesh,
    get_adjacency_list,
    load_object_data,
    transform_points_world_to_local,
)

# 21 robot keypoints per hand: palm (= wrist), then finger{1..5}_link{1..4}.
# NB the legacy labels "link1..4 = MCP/PIP/DIP/tip" are OFF BY ONE: measured + confirmed against the
# upstream wuji-retargeting repo, the real anatomy is link1(~link2)=MCP, link3=PIP, link4=DIP, and the
# TIP is the tip_link geom (link1->link2 is only 0.46cm = the two stacked knuckle joints). Use
# --remap-fingers (hand_body_names(remap=True)) for the anatomically-correct [link1,link3,link4,tip].
HAND_BODIES = {
    "left": ["wj_left_palm_link"] + [f"wj_left_finger{i}_link{L}" for i in (1, 2, 3, 4, 5) for L in (1, 2, 3, 4)],
    "right": ["wjr_right_palm_link"] + [f"wjr_right_finger{i}_link{L}" for i in (1, 2, 3, 4, 5) for L in (1, 2, 3, 4)],
}
SIDE_PREFIX = {"left": "wj_left", "right": "wjr_right"}


# Finger keypoint modes -> (robot link suffixes per finger, human-kp relative indices per finger).
# Human 21-kp order: 0=wrist, then per finger [MCP,PIP,DIP,tip] at 1+(i-1)*4+[0,1,2,3].
#   full    -- legacy [link1,link2,link3,link4] <-> [MCP,PIP,DIP,tip]  (OFF BY ONE, see HAND_BODIES note)
#   remap   -- [link1,link3,link4,link4] <-> [MCP,PIP,DIP,TIP]         (anatomical; last=TIP->tip_link)
#   mcp_tip -- [link1,link4] <-> [MCP,TIP] only (drop PIP+DIP; no E_bone -- needs >=3 kp/finger)
FINGER_KP = {
    "full":    ((1, 2, 3, 4), (0, 1, 2, 3)),
    "remap":   ((1, 3, 4, 4), (0, 1, 2, 3)),
    "mcp_tip": ((1, 4),       (0, 3)),
}


def hand_body_names(side, mode="full"):
    """The keypoint body names for a finger-kp mode (the last link per finger is the TIP slot)."""
    pfx = SIDE_PREFIX[side]
    links = FINGER_KP[mode][0]
    return [f"{pfx}_palm_link"] + [f"{pfx}_finger{i}_link{L}" for i in (1, 2, 3, 4, 5) for L in links]


def human_kp_indices(mode="full"):
    """Which of the 21 human keypoints to keep for a mode (to match hand_body_names index-for-index)."""
    rel = FINGER_KP[mode][1]
    return [0] + [1 + (i - 1) * 4 + r for i in (1, 2, 3, 4, 5) for r in rel]


def _kp(i, L):
    """Keypoint index for finger i (1..5), link L (1..4) in the 21-keypoint order (0 = palm)."""
    return 1 + (i - 1) * 4 + (L - 1)


TIP_KP = [4, 8, 12, 16, 20]                     # the 5 fingertip rows in the 21-keypoint interaction mesh


# Adjacent finger-bone pairs (a1->b1, a2->b2) for E_bone: consecutive bones sharing a joint, per
# finger -> (MCP->PIP, PIP->DIP) and (PIP->DIP, DIP->tip). 2 pairs x 5 fingers = 10.
BONE_PAIRS = [(_kp(i, 1), _kp(i, 2), _kp(i, 2), _kp(i, 3)) for i in (1, 2, 3, 4, 5)] \
    + [(_kp(i, 2), _kp(i, 3), _kp(i, 3), _kp(i, 4)) for i in (1, 2, 3, 4, 5)]


def wrist_frame(P, mcp):
    """Right-handed WRIST frame from keypoints P (rows, any consistent frame); mcp = the 5 MCP indices
    [thumb..pinky]. Columns are the axes: x = wrist->middle MCP (down the hand), n = palm normal,
    y = n x x. Estimated identically for robot and human so their bone directions are comparable.
    To express an object/world vector v in this frame use R.T @ v (bone_mode='wrist')."""
    w = P[0]
    x = P[mcp[2]] - w; x = x / (np.linalg.norm(x) + 1e-9)         # wrist -> middle MCP
    v = P[mcp[1]] - w                                             # wrist -> index MCP
    n = np.cross(x, v); n = n / (np.linalg.norm(n) + 1e-9)        # palm normal
    y = np.cross(n, x)
    return np.column_stack([x, y, n])


def so3_log(R):
    """Log map SO(3)->R^3: the rotation vector (axis*angle) of R. Used for geodesic residuals
    e_R = log(R_ref^T R). Numerically robust near 0 and pi."""
    cos = (np.trace(R) - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    theta = np.arccos(cos)
    if theta < 1e-7:                                                  # ~identity: first-order skew part
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    if np.pi - theta < 1e-6:                                          # near pi: axis from (R+I) columns
        A = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k] / np.sqrt(max(A[k, k], 1e-12))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * theta
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (theta / (2.0 * np.sin(theta)))


def exp_laplacian_matrix(vertices, adj_list, kappa=30.0, uniform=False, epsilon=1e-9):
    """Row-normalized Laplacian operator L (N x N), Delta = L @ vertices, Delta_i = v_i - Σ_j w_ij v_j.

    Weights per neighbour:
        --exp-weight (uniform=False): w_ij = softmax_j exp(-kappa*||v_i - v_j||)   (TopoRetarget Eq.5)
        uniform (uniform=True):       w_ij = 1 / deg(i)                            (OmniRetarget)

    Args:
        vertices (np.ndarray): (N, 3) interaction-mesh vertices (source geometry the weights use).
        adj_list (list[list[int]]): neighbour indices per vertex.
        kappa (float): exponential spatial decay (30 in the paper).
        uniform (bool): fall back to uniform weights (for the OmniRetarget-style baseline).

    Returns:
        np.ndarray: (N, N) Laplacian matrix.
    """
    n = len(vertices)
    L = np.eye(n)
    for i in range(n):
        nbr = adj_list[i]
        if not len(nbr):
            continue
        if uniform:
            w = np.ones(len(nbr)) / len(nbr)
        else:
            d = np.linalg.norm(vertices[i] - vertices[nbr], axis=1)
            w = np.exp(-kappa * d)
            w = w / (w.sum() + epsilon)
        for j, wj in zip(nbr, w):
            L[i, j] -= wj
    return L


class TopoRetargeter:
    """TopoRetarget-style finger+wrist retargeter (Stage B), OmniRetarget interaction mesh + upgrades.

    Per SQP iteration the problem minimizes, over the wrist+finger step dq:
        1. [Cost] E_IM   -- interaction-mesh Laplacian match (exp-weighted if --exp-weight).
        2. [Cost] E_bone -- relative finger-bone direction match (if --bone), linearized in dq.
        3. [Cost] Stage-A regularizer on the freed wrist DOF.
        4. [Constraint] joint limits.
        5. [Constraint] trust region on dq.
        6. [Constraint] slack object non-penetration (--soft-pen), velocity cap (--vel-bound),
           finger-finger (--self-col); + contact attraction cost (--contact).
    """

    def __init__(self, model, side, object_points_local, kappa=30.0, exp_weight=True, use_bone=True,
                 w_contact=0.0, vel_bound=False, vmax_trans=0.03, vmax_rot=0.15, self_col=False,
                 surface_pq=None, reach_r=0.05,
                 w_bone=0.1, wrist_reg=0.0, sqp_iters=15, step=0.3,
                 use_smooth=False, w_smooth=0.05, use_warmup=False, warmup_iters=8,
                 soft_pen=False, obj_name="", tip_mode="joint", w_tip=1.0, w_dir=0.0, finger_kp="full",
                 lambda_im=1.0, im_normalize=False, bone_mode="cosine", warmup_w_bone=None,
                 warmup_w_smooth=None, base_reg_trans=0.0, base_reg_rot=0.0, sqp_tol=0.0,
                 pen_coverage="full", pen_tau=0.001, pen_emergency=False, base_joints=None, base_bound=None,
                 base_manifold=False, continuity_fallback=False):
        """
        Args:
            model (mujoco.MjModel): the full Wuji model (fingers free).
            side (str): "left" or "right".
            object_points_local (np.ndarray): (P, 3) object surface points in the object frame.
            kappa (float): exponential Laplacian decay (TopoRetarget: 30).
            exp_weight (bool): exp-weighted Laplacian (True) vs uniform (False, OmniRetarget baseline).
            use_bone (bool): add the E_bone bone-direction term.
            w_bone (float): weight of E_bone relative to E_IM.
            wrist_reg (float): weight pulling the freed wrist DOF toward the Stage-A pose.
            sqp_iters (int): SQP iterations per frame.
            step (float): trust-region cap on ||dq|| per iteration.
        """
        self.model = model
        self.data = mujoco.MjData(model)
        self.O = object_points_local
        self.O_target = None                                        # SOURCE/TARGET SEP: full-size target object pts
        #   (default None => robot Laplacian uses self.O; set to O_full to build V_r on the full-size target).
        self.kappa = kappa
        self.exp_weight = exp_weight
        self.w_tip = w_tip                                           # extra weight on the 5 fingertip E_IM rows
        # paper-faithful knobs (TopoRetarget arXiv:2606.16272). Defaults reproduce the current baseline.
        self.lambda_im = lambda_im                                  # E_IM weight (paper lambda_IM=500)
        self.lambda_im_sched = None            # per-frame contact-scheduled E_IM weight (hierarchical arm refine)
        self.im_normalize = im_normalize                            # divide E_IM by N_v (# mesh vertices)
        self.bone_mode = bone_mode                                  # "cosine" (curl) | "direction" (paper diff)
        self.warmup_w_bone = w_bone if warmup_w_bone is None else warmup_w_bone      # paper warmup bone=1.0
        self.warmup_w_smooth = w_smooth if warmup_w_smooth is None else warmup_w_smooth  # paper warmup smooth=2.5
        self.base_reg_trans = base_reg_trans                        # base-pose reg vs PREVIOUS frame (paper trans=100)
        self.base_reg_rot = base_reg_rot                            # base-pose reg vs PREVIOUS frame (paper rot=1)
        self.base_reg_rot_sched = None                             # optional (T,) per-frame override of base_reg_rot
        self._cur_t = 0                                            #   (distance-gated); None -> use the scalar above
        self.sqp_tol = sqp_tol                                      # >0: iterate until ||dq||<tol (else fixed sqp_iters)
        self.use_bone = use_bone
        self.w_bone = w_bone
        self.wrist_reg = wrist_reg
        self.wrist_reg_sched = None            # per-frame contact-gated wrist->q_ref anchor (elbow-aware non-contact stab)
        self.finger_reg_sched = None           # per-frame finger-return weight (0 contact -> W off-contact)
        self.q_finger_ref = None               # finger reference pose (neutral/open) for the finger qadr
        self.base_bound = base_bound                                 # HARD cap: |freed base DOF - Stage A| <= base_bound (rad)
        self.sqp_iters = sqp_iters
        self.step = step
        self.use_smooth = use_smooth                                 # E_reg: temporal smoothness (kills glitch)
        self.w_smooth = w_smooth
        self.use_warmup = use_warmup                                 # TopoRetarget per-frame E_bone warmup init
        self.warmup_iters = warmup_iters
        # contact attraction: pull each fingertip to the HUMAN's surface contact -- the only term encoding
        # contact TOPOLOGY (thumb on rim, fingers on top), which Laplacian + soft-pen cannot. Per-frame
        # targets stashed on self._contact by retarget_fingers.
        self.w_contact = w_contact
        self._contact = None                                         # (tgt_loc (5,3), weight (5,)) per frame
        # hard velocity bound (REGRIND): |q_t - q_{t-1}| <= vmax -- a rate cap that kills contact-flip jumps
        # without the freeze a large w_smooth caused. Per-DOF vmax built below once fin_j exists.
        self.vel_bound = vel_bound
        self._vmax_trans, self._vmax_rot = vmax_trans, vmax_rot
        self.self_col = self_col                                     # finger-finger non-penetration (TeleDexter L_col)
        # reachable targets (TeleDexter L_surf): chase the surface point nearest the ROBOT tip if within
        # reach_r of the human contact -- reachable by construction, so no tip hyperextension.
        self.surface_pq = surface_pq
        self.reach_r = reach_r

        # finger keypoint mode (FINGER_KP). full=legacy off-by-one; remap=anatomical [MCP,PIP,DIP,TIP];
        # mcp_tip=[MCP,TIP] only. The LAST link per finger is the TIP slot; for non-full modes it is link4,
        # which MUST be relocated to the tip_link geom or it coincides with the DIP slot -> force tip_mode.
        self.finger_kp = finger_kp
        self.k_per_finger = len(FINGER_KP[finger_kp][0])
        self.tip_kp = [1 + (fi - 1) * self.k_per_finger + (self.k_per_finger - 1) for fi in range(1, 6)]
        self.mcp_kp = [1 + (fi - 1) * self.k_per_finger for fi in range(1, 6)]   # first (MCP) kp of each finger
        self.w_dir = w_dir                                                       # E_direction: match MCP->TIP dir
        if finger_kp != "full" and tip_mode == "joint":
            tip_mode = "geom"
            print(f"    [{side}] finger_kp='{finger_kp}': TIP slot forced to tip_mode='geom' (else TIP==DIP=link4)")
        if self.k_per_finger < 3 and use_bone:                       # E_bone needs >=3 kp/finger (a bend angle)
            use_bone = False
            print(f"    [{side}] finger_kp='{finger_kp}': E_bone disabled ({self.k_per_finger} kp/finger < 3)")
        self.use_bone = use_bone
        self.kb = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in hand_body_names(side, finger_kp)]
        self.nk = len(self.kb)                                       # 1 palm + 5 * k_per_finger

        # tip keypoint source (ablation): the 5 tip keypoints are finger link4 BODY ORIGINS = the DIP
        # joint, which sits 2-3 cm SHORT of the real fingertip. The model already carries the tip as a
        # *_tip_link geom on link4, so the true tip is model-derived (no hand-tuned offset):
        #   "joint" -- link4 origin (default; reproduces every prior run)
        #   "geom"  -- the tip_link collision geom ORIGIN (~1.8 cm fingers / 2.4 cm thumb out)
        #   "pad"   -- the FARTHEST vertex of that geom's mesh (the physical pad, ~2.9-3.3 cm out)
        # Stored as a constant offset in the link4 body frame; solve_single_iteration relocates the
        # keypoint AND evaluates its Jacobian at the SAME world point (else the linearization is inconsistent).
        self.tip_mode = tip_mode
        self.tip_local = {}                                          # {kp_index: offset in link4 frame}
        if tip_mode != "joint":
            for fi in range(1, 6):
                kpi = self.tip_kp[fi - 1]; b4 = self.kb[kpi]         # TIP slot (mode-aware) = a link4 body
                tg = next((g for g in range(model.ngeom)
                           if int(model.geom_bodyid[g]) == b4 and int(model.geom_dataid[g]) >= 0
                           and f"finger{fi}_tip_link" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH,
                                                          int(model.geom_dataid[g])) or "")
                           and (model.geom_contype[g] or model.geom_conaffinity[g])), None)
                if tg is None:
                    continue
                loc = np.array(model.geom_pos[tg], float)           # tip geom origin, in link4 frame
                if tip_mode == "pad":
                    did = int(model.geom_dataid[tg])
                    V = model.mesh_vert[model.mesh_vertadr[did]:model.mesh_vertadr[did] + model.mesh_vertnum[did]]
                    Rg = trimesh.transformations.quaternion_matrix(model.geom_quat[tg])[:3, :3]
                    Vb = loc + V @ Rg.T                             # tip-mesh verts in link4 frame
                    loc = Vb[np.linalg.norm(Vb, axis=1).argmax()]  # farthest = the pad
                self.tip_local[kpi] = loc
            print(f"    [{side}] tip keypoints -> '{tip_mode}' ({len(self.tip_local)}/5 fingers, "
                  f"mean offset {100 * np.mean([np.linalg.norm(v) for v in self.tip_local.values()]):.1f} cm)")
        pfx = SIDE_PREFIX[side]
        finger_j = [j for j in range(model.njnt)
                    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(pfx)
                    and "finger" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "")]
        # free base DOF by name, skipping absent ones: the G1 arm contributes roll/pitch/yaw (3, the
        # original free-wrist), a FLOATING-hand model (make_hand_model.py) additionally names x/y/z
        # slide joints so the same solver places the hand in space -- 6 base DOF, no G1 change.
        if base_joints is None:                                      # default: the named wrist DOF (6 floating / 3 G1)
            wrist_j = [j for j in (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wrist_{a}_joint")
                                   for a in ("x", "y", "z", "roll", "pitch", "yaw")) if j >= 0]
        else:                                                        # explicit base set: [] = fingers-only (B1),
            wrist_j = [j for j in (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)  # wrist joints (B2), arm (B3)
                                   for nm in base_joints) if j >= 0]
        fin_j = wrist_j + finger_j                                   # free wrist (base) leads, then fingers
        self.n_wrist = len(wrist_j)
        self.reg_idx = list(range(self.n_wrist))                     # wrist DOF regularized toward Stage A
        self.qadr = [int(model.jnt_qposadr[j]) for j in fin_j]
        self.dadr = [int(model.jnt_dofadr[j]) for j in fin_j]
        self.lo = np.array([model.jnt_range[j][0] for j in fin_j])
        self.hi = np.array([model.jnt_range[j][1] for j in fin_j])
        self.nm = len(fin_j)                                         # 3 wrist + 20 finger
        # per-DOF velocity cap: slides (the floating root's x/y/z) in metres, hinges in radians
        self.vmax = np.array([self._vmax_trans if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE
                              else self._vmax_rot for j in fin_j])

        # --- S3 (manifold) base: 3 slides + 1 BALL joint. Optimizer works in the 6D tangent [dx, dtheta];
        #     qpos stores a quaternion; SQP steps applied via mj_integratePos; prior/smooth use log-map residuals.
        self.base_manifold = base_manifold
        self.continuity_fallback = continuity_fallback                              # rerun-and-select isolated glitch frames
        self._status_counts = {}                                                   # QP status histogram (main iters)
        if base_manifold:
            assert base_bound is None, "base_bound (hard Stage-A box) not implemented for the manifold base"
            # vel_bound IS implemented for the manifold (log-map rate cap): see the manifold constraints block.
            slides = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wrist_{a}_joint") for a in ("x", "y", "z")]
            ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wrist_ball_joint")
            assert ball >= 0 and all(s >= 0 for s in slides), "base_manifold model needs 3 slides + a ball joint"
            ba = int(model.jnt_dofadr[ball])
            self.dadr = ([int(model.jnt_dofadr[s]) for s in slides] + [ba, ba + 1, ba + 2]
                         + [int(model.jnt_dofadr[j]) for j in finger_j])           # qvel (6 base + 20 finger)
            self.nm = len(self.dadr); self.n_wrist = 6; self.reg_idx = list(range(6))
            self._trans_dq = [0, 1, 2]; self._rot_dq = [3, 4, 5]                    # slots in dq
            self._lin_dq = [0, 1, 2] + list(range(6, self.nm))                      # slides + fingers (linear)
            self._trans_qadr = [int(model.jnt_qposadr[s]) for s in slides]
            self._fin_qadr = [int(model.jnt_qposadr[j]) for j in finger_j]
            self._lin_qadr = self._trans_qadr + self._fin_qadr                      # aligns with self._lin_dq
            self._ball_qadr = int(model.jnt_qposadr[ball])                          # start of the 4-quat
            self.lo = np.array([model.jnt_range[s][0] for s in slides] + [model.jnt_range[j][0] for j in finger_j])
            self.hi = np.array([model.jnt_range[s][1] for s in slides] + [model.jnt_range[j][1] for j in finger_j])
            self._base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"hand_root_{side}")
            self._R_ref = self._trans_ref = self._R_prev = self._lin_prev = None    # set per-frame by the loop

        if self_col:
            # finger-finger pairs (TeleDexter L_col, mesh-accurate via mj_geomDistance): link3/link4 of
            # ADJACENT fingers -- where the ring-pinky intersections happened.
            link_geom = {}
            for g in range(model.ngeom):
                bb = int(model.geom_bodyid[g])
                nm_ = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bb) or ""
                if nm_.startswith(pfx) and "finger" in nm_ and nm_[-1] in "34":
                    link_geom.setdefault(nm_, (g, bb))               # first geom of each link body
            self.ff_pairs = []
            for fa in range(1, 5):                                   # adjacent fingers only
                for la in "34":
                    for lb in "34":
                        a = link_geom.get(f"{pfx}_finger{fa}_link{la}")
                        b = link_geom.get(f"{pfx}_finger{fa + 1}_link{lb}")
                        if a and b:
                            self.ff_pairs.append((*a, *b))
            self.w_s = getattr(self, "w_s", 1e5)                     # slack penalty even without soft_pen
            print(f"    [{side}] finger-finger non-penetration on {len(self.ff_pairs)} link pairs")

        # slack-based object non-penetration (TopoRetarget Eq.8): phi + s >= -tau (soft), phi >= -b
        # (hard), slack s in [0, b-tau] penalized -- NEVER infeasible. Object geoms = CoACD pieces if
        # the model has them (decompose), else the single hull geom.
        self.soft_pen = soft_pen
        if soft_pen:
            self.obj_geoms = [g for g in range(model.ngeom)
                              if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(f"{obj_name}_piece")]
            if not self.obj_geoms:
                ob = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{obj_name}_link")
                self.obj_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == ob]
            self.finger_geoms = []
            if pen_coverage == "partial":                            # DIAGNOSTIC: reproduce the OLD coverage --
                seen = set()                                         # one geom per finger-link body, no palm, no
                for g in range(model.ngeom):                         # separate tip_link geom (regression 2x2 test).
                    bb = int(model.geom_bodyid[g])
                    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bb) or ""
                    if nm.startswith(pfx) and "finger" in nm and bb not in seen:
                        self.finger_geoms.append((g, bb)); seen.add(bb)
            else:                                                    # COLLISION geoms, selectable by GROUP (Track 2):
                want_palm = pen_coverage in ("full", "links_palm")   #   links | links_palm | links_tips | full
                want_tip = pen_coverage in ("full", "links_tips")    # a geom is palm / fingertip (tip_link mesh) / link
                for g in range(model.ngeom):
                    bb = int(model.geom_bodyid[g])
                    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bb) or ""
                    if not (nm.startswith(pfx) and (model.geom_contype[g] or model.geom_conaffinity[g])):
                        continue
                    msh = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g])
                           if model.geom_dataid[g] >= 0 else "") or ""
                    is_palm, is_tip = nm.endswith("palm_link"), msh.endswith("tip_link")
                    if is_palm and not want_palm:
                        continue
                    if is_tip and not want_tip:
                        continue
                    self.finger_geoms.append((g, bb))                # links always in; palm/tip gated by the group
            self.pen_detect, self.pen_b, self.w_s = 0.1, 0.03, 1e5
            self.pen_tau = pen_tau
            self.pen_emergency = pen_emergency          # unbounded slack -> QP never infeasible from non-pen
            self._pen_slack_max = 0.0                   # max object non-pen slack of the LAST solve (for tracking)
            print(f"    [{side}] soft non-pen: {len(self.finger_geoms)} hand geom(s) [{pen_coverage}, tau={self.pen_tau*1000:g}mm]"
                  f" vs {len(self.obj_geoms)} object geom(s) ({'CoACD pieces' if len(self.obj_geoms) > 1 else 'single hull'})")

    def solve_single_iteration(self, qfull, q_move, Lk, lap_src, lapO, quat, trans, bone_src, q_ref,
                               q_prev=None, warmup=False, dir_src=None):
        """One SQP step: linearize the interaction-mesh Laplacian (+ bone term) and solve a QP for dq.

        Args:
            qfull (np.ndarray): full qpos with frozen body/object for this frame.
            q_move (np.ndarray): current wrist+finger angles (nm,).
            Lk (np.ndarray): keypoint block of the Laplacian, (n, nk).
            lap_src (np.ndarray): target Laplacian coordinates, (n, 3).
            lapO (np.ndarray): fixed object contribution to the Laplacian, (n, 3).
            quat, trans (np.ndarray): object pose (wxyz, xyz) for the object-local frame.
            bone_src (np.ndarray): (len(BONE_PAIRS),) human cos(bend) = d1_s . d2_s per adjacent pair.
            q_ref (np.ndarray): Stage-A pose of the optimized DOF (nm,); wrist regularizer target.

        Returns:
            np.ndarray | None: the joint step dq (nm,), or None if the QP failed.
        """
        if not self.base_manifold:
            qfull[self.qadr] = q_move                                # Euler: q_move IS the optimized qpos slice
        # manifold: qfull is the authoritative state (integrated via mj_integratePos by the loop); use it as-is
        self.data.qpos[:] = qfull
        mujoco.mj_forward(self.model, self.data)
        Rt = trimesh.transformations.quaternion_matrix(quat)[:3, :3]
        pk_world = np.array([self.data.xpos[b] for b in self.kb])
        for kpi, loc in self.tip_local.items():                                  # relocate tips to the real pad/geom
            b4 = self.kb[kpi]
            pk_world[kpi] = self.data.xpos[b4] + self.data.xmat[b4].reshape(3, 3) @ loc
        pk_loc = transform_points_world_to_local(quat, trans, pk_world)          # robot keypoints, object-local
        J = np.zeros((self.nk, 3, self.nm))
        for i, b in enumerate(self.kb):
            jp = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jp, None, pk_world[i], b)        # SAME point as the position (tip-consistent)
            J[i] = Rt.T @ jp[:, self.dadr]                                        # object-local Jacobian (3, nm)

        dq = cp.Variable(self.nm)
        # warmup solves E_bone + smoothness only (no Laplacian) to INITIALIZE the frame (TopoRetarget Eq.2)
        obj_terms = []
        if not warmup:
            dpk = cp.vstack([J[i] @ dq for i in range(self.nk)])
            lap = Lk @ (pk_loc + dpk) + lapO                                     # linearized Laplacian
            _lam = float(self.lambda_im_sched[self._cur_t]) if self.lambda_im_sched is not None else self.lambda_im
            im_scale = _lam / (lap_src.shape[0] if self.im_normalize else 1.0)  # paper: 500/N_v (contact-scheduled)
            if self.w_tip != 1.0:                                               # weight the 5 fingertip rows up
                aw = np.ones((lap_src.shape[0], 3)); aw[self.tip_kp] = self.w_tip  # per-row amplitude^2
                obj_terms.append(im_scale * cp.sum_squares(cp.multiply(np.sqrt(aw), lap - lap_src)))
            else:
                obj_terms.append(im_scale * cp.sum_squares(lap - lap_src))

        if self.use_bone:                                                        # E_bone
            wb = self.warmup_w_bone if warmup else self.w_bone
            Rw = wrist_frame(pk_loc, self.mcp_kp) if self.bone_mode == "wrist" else None  # robot wrist frame this iter
            for pi, (a1, b1, a2, b2) in enumerate(BONE_PAIRS):
                v1 = pk_loc[b1] - pk_loc[a1]; n1 = max(np.linalg.norm(v1), 1e-6); d1 = v1 / n1
                v2 = pk_loc[b2] - pk_loc[a2]; n2 = max(np.linalg.norm(v2), 1e-6); d2 = v2 / n2
                Jd1 = (np.eye(3) - np.outer(d1, d1)) / n1 @ (J[b1] - J[a1])      # d(d1)/dq  (3, nm)
                Jd2 = (np.eye(3) - np.outer(d2, d2)) / n2 @ (J[b2] - J[a2])      # d(d2)/dq
                Jd1[:, :self.n_wrist] = 0.0; Jd2[:, :self.n_wrist] = 0.0         # FINGER DOF only
                if self.bone_mode == "cosine":                                  # cosine bend (curl, rotation-invariant)
                    grad = d2 @ Jd1 + d1 @ Jd2                                   # d(d1.d2)/dq  (nm,)
                    resid = (float(d1 @ d2) - bone_src[pi]) + grad @ dq          # scalar, affine in dq
                    obj_terms.append(wb * cp.square(resid))
                else:                                                            # PAPER dir-diff ||(d1-d2)r-(d1-d2)h||^2
                    dd, Jdd = (d1 - d2), (Jd1 - Jd2)                             # object frame
                    if self.bone_mode == "wrist":                               # -> express in the robot WRIST frame
                        dd, Jdd = Rw.T @ dd, Rw.T @ Jdd                          # (bone_src is in the human wrist frame)
                    resid = (dd - bone_src[pi]) + Jdd @ dq                       # (3,) vector, affine in dq
                    obj_terms.append(wb * cp.sum_squares(resid))

        if self.w_dir > 0 and dir_src is not None and not warmup:                # E_direction: match the whole
            for fi in range(5):                                                  # finger's MCP->TIP unit direction
                a, b = self.mcp_kp[fi], self.tip_kp[fi]                           # (works with only 2 kp/finger)
                v = pk_loc[b] - pk_loc[a]; nv = max(np.linalg.norm(v), 1e-6); dh = v / nv
                Jd = (np.eye(3) - np.outer(dh, dh)) / nv @ (J[b] - J[a])          # d(dhat)/dq  (3, nm)
                Jd[:, :self.n_wrist] = 0.0                                        # finger DOF only (wrist places the hand)
                resid = (dh - dir_src[fi]) + Jd @ dq                             # (3,) affine in dq
                obj_terms.append(self.w_dir * cp.sum_squares(resid))

        if self.w_contact > 0 and self._contact is not None and not warmup:      # contact attraction
            tgt_loc, wgt = self._contact                              # wgt: 0..1 ramped weight per finger
            for fi in range(5):
                if wgt[fi] < 1e-3:
                    continue                                          # only frames the HUMAN touches
                tip = (fi + 1) * 4                                    # keypoint index of finger fi's tip
                tgt = tgt_loc[fi]
                if self.surface_pq is not None:                       # reachable target (TeleDexter L_surf)
                    near, _, _ = self.surface_pq.on_surface(pk_loc[tip][None])
                    if np.linalg.norm(near[0] - tgt) <= self.reach_r:
                        tgt = near[0]                                 # same contact REGION, robot-reachable point
                obj_terms.append(self.w_contact * float(wgt[fi]) *
                                 cp.sum_squares(pk_loc[tip] + J[tip] @ dq - tgt))

        if self.base_manifold:                                                   # S3: log-map smoothness + prior
            Rc = self.data.xmat[self._base_body].reshape(3, 3)                    # current base orientation
            lin_cur = qfull[self._lin_qadr]                                       # slides + fingers (linear DoF)
            if self.use_smooth and self._R_prev is not None:
                ws = self.warmup_w_smooth if warmup else self.w_smooth
                eS = so3_log(self._R_prev.T @ Rc)                                 # geodesic frame-to-frame rotation
                obj_terms.append(ws * cp.sum_squares(eS + dq[self._rot_dq]))      # rotation smoothness (log residual)
                obj_terms.append(ws * cp.sum_squares((lin_cur - self._lin_prev) + dq[self._lin_dq]))
            if not warmup:                                                       # translation + rotation priors INDEPENDENT
                if self.base_reg_trans > 0:
                    obj_terms.append(self.base_reg_trans *
                                     cp.sum_squares((qfull[self._trans_qadr] - self._trans_ref) + dq[self._trans_dq]))
                rot_w = self.base_reg_rot if self.base_reg_rot_sched is None else float(self.base_reg_rot_sched[self._cur_t])
                if rot_w > 0:
                    eR = so3_log(self._R_ref.T @ Rc)                             # geodesic deviation from Kabsch ref
                    obj_terms.append(rot_w * cp.sum_squares(eR + dq[self._rot_dq]))
        else:
            if self.use_smooth and q_prev is not None:                               # E_reg: temporal smoothness
                ws = self.warmup_w_smooth if warmup else self.w_smooth
                obj_terms.append(ws * cp.sum_squares((q_move + dq) - q_prev))
            _wr = float(self.wrist_reg_sched[self._cur_t]) if self.wrist_reg_sched is not None else self.wrist_reg
            if self.reg_idx and _wr > 0:                                             # combined wrist reg -> q_ref (contact-gated if sched)
                ri = self.reg_idx
                obj_terms.append(_wr * cp.sum_squares((q_move[ri] + dq[ri]) - q_ref[ri]))
            _fr = float(self.finger_reg_sched[self._cur_t]) if self.finger_reg_sched is not None else 0.0
            if _fr > 0 and self.q_finger_ref is not None and not warmup:             # phase-aware finger-return -> neutral/open
                fi = list(range(self.n_wrist, self.nm))
                obj_terms.append(_fr * cp.sum_squares((q_move[fi] + dq[fi]) - self.q_finger_ref))
            if self.base_reg_trans > 0 and not warmup:                               # PAPER base-pose PRIOR (Eq.9), FINAL
                ti = list(range(3)); rri = list(range(3, self.n_wrist))              # objective ONLY (warmup Eq.2 = bone+smooth).
                # ADAPTATION: paper writes ||q_base||^2; we apply its weights to deviation from the per-frame KABSCH base.
                obj_terms.append(self.base_reg_trans * cp.sum_squares((q_move[ti] + dq[ti]) - q_ref[ti]))
                if rri:
                    rot_w = self.base_reg_rot if self.base_reg_rot_sched is None \
                        else float(self.base_reg_rot_sched[self._cur_t])   # distance-gated per-frame rotation prior
                    obj_terms.append(rot_w * cp.sum_squares((q_move[rri] + dq[rri]) - q_ref[rri]))

        if not obj_terms:                                                        # nothing to optimize (e.g. warmup w/o bone)
            return np.zeros(self.nm)
        if self.base_manifold:                                                   # box only on linear DoF (ball is free)
            lin_cur = qfull[self._lin_qadr]
            cons = [lin_cur + dq[self._lin_dq] >= self.lo, lin_cur + dq[self._lin_dq] <= self.hi, cp.norm(dq) <= self.step]
            if self.vel_bound and self._R_prev is not None and not warmup:        # MANIFOLD rate cap (log-map)
                Rc = self.data.xmat[self._base_body].reshape(3, 3)               # current base orientation
                vlin = np.array([self._vmax_trans] * 3 + [self._vmax_rot] * (len(self._lin_dq) - 3))  # slides|fingers
                cons += [(lin_cur + dq[self._lin_dq]) - self._lin_prev <= vlin,   # translation + finger rate cap
                         (lin_cur + dq[self._lin_dq]) - self._lin_prev >= -vlin]
                eS = so3_log(self._R_prev.T @ Rc)                                 # rotation rate = ||log(R_prev^T R_new)||
                cons += [(eS + dq[self._rot_dq]) <= self._vmax_rot, (eS + dq[self._rot_dq]) >= -self._vmax_rot]
        else:
            cons = [q_move + dq >= self.lo, q_move + dq <= self.hi, cp.norm(dq) <= self.step]
        if self.base_bound is not None and self.reg_idx:                          # HARD Stage-A box on the freed base DOF
            ri = self.reg_idx                                                    # (guarantees B2/B3 stay near A, not just soft)
            cons += [(q_move[ri] + dq[ri]) - q_ref[ri] <= self.base_bound,
                     (q_move[ri] + dq[ri]) - q_ref[ri] >= -self.base_bound]
        if self.vel_bound and q_prev is not None:                    # REGRIND-style hard rate cap
            cons += [(q_move + dq) - q_prev <= self.vmax, (q_move + dq) - q_prev >= -self.vmax]
        if self.soft_pen and not warmup:                                         # slack object non-penetration
            ft = np.zeros(6)
            slacks = []
            for g, bb in self.finger_geoms:
                for og in self.obj_geoms:
                    dist = mujoco.mj_geomDistance(self.model, self.data, g, og, self.pen_detect, ft)
                    if dist < self.pen_detect:
                        nrm = ft[:3] - ft[3:]
                        nn = np.linalg.norm(nrm)
                        if nn < 1e-9:
                            continue
                        jp = np.zeros((3, self.model.nv))
                        mujoco.mj_jac(self.model, self.data, jp, None, ft[:3], bb)
                        phi = dist + (np.sign(dist) * (nrm / nn) @ jp[:, self.dadr]) @ dq   # linearized signed dist
                        s = cp.Variable(nonneg=True)
                        if self.pen_emergency:                       # EMERGENCY slack: unbounded, so the QP can
                            cons += [phi + s >= -self.pen_tau]       # accept a big violation rather than return None
                        else:                                        # (the w_s penalty still resists penetration). The
                            cons += [phi >= -self.pen_b,             # default hard bound can go INFEASIBLE when a moving
                                     phi + s >= -self.pen_tau,       # object drives an already-penetrating geom past
                                     s <= self.pen_b - self.pen_tau]  # pen_b in one step -> crash + frozen pose.
                        slacks.append(s)
            if slacks:
                obj_terms.append((self.w_s / 2) * cp.sum_squares(cp.hstack(slacks)))
        if self.self_col and not warmup:                             # finger-finger non-penetration
            # keep adjacent-finger links >= 3 mm apart, slacked (never infeasible). RELATIVE Jacobian
            # (Ja - Jb): both links move, unlike the static object.
            ft = np.zeros(6)
            ff_slacks = []
            for ga, ba, gb, bb2 in self.ff_pairs:
                dist = mujoco.mj_geomDistance(self.model, self.data, ga, gb, 0.05, ft)
                if dist < 0.05:
                    nrm = ft[:3] - ft[3:]
                    nn = np.linalg.norm(nrm)
                    if nn < 1e-9:
                        continue
                    ja = np.zeros((3, self.model.nv)); jb = np.zeros((3, self.model.nv))
                    mujoco.mj_jac(self.model, self.data, ja, None, ft[:3], ba)
                    mujoco.mj_jac(self.model, self.data, jb, None, ft[3:], bb2)
                    phi = dist + (np.sign(dist) * (nrm / nn) @ (ja - jb)[:, self.dadr]) @ dq
                    s = cp.Variable(nonneg=True)
                    cons += [phi + s >= 0.003]
                    ff_slacks.append(s)
            if ff_slacks:
                obj_terms.append((self.w_s / 2) * cp.sum_squares(cp.hstack(ff_slacks)))
        prob = cp.Problem(cp.Minimize(cp.sum(obj_terms)), cons)
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            return None
        if not warmup:
            self._status_counts[prob.status] = self._status_counts.get(prob.status, 0) + 1
            self._last_status = prob.status                                       # for the guarded-SQP quality check
        if getattr(self, "_dbg_t", -1) == self._cur_t and not warmup and dq.value is not None:
            nd = float(np.linalg.norm(dq.value))
            self._dbg.append(dict(status=prob.status, obj=float(prob.value) if prob.value is not None else None,
                                  ndq=nd, trust_active=nd >= 0.99 * self.step))
        self._pen_slack_max = (max((float(s.value) for s in slacks if s.value is not None), default=0.0)
                               if (self.soft_pen and not warmup and slacks) else 0.0)
        return dq.value

    def _frame_inputs(self, t, hkp, obj_quat, obj_pos, n, contact_tgt, contact_mask):
        """Recompute a single frame's interaction-mesh + bone/dir inputs (same as the main loop)."""
        self._contact = (contact_tgt[t], contact_mask[t]) if contact_tgt is not None else None
        hloc = transform_points_world_to_local(obj_quat[t], obj_pos[t], hkp[t])
        src = np.vstack([hloc, self.O])
        _, tets = create_interaction_mesh(src)
        adj = get_adjacency_list(tets, n)
        L = exp_laplacian_matrix(src, adj, kappa=self.kappa, uniform=not self.exp_weight)
        lap_src = L @ src
        Lk, lapO = L[:, :self.nk], L[:, self.nk:] @ self.O
        vec_mode = self.bone_mode in ("direction", "wrist")
        bone_src = np.zeros((len(BONE_PAIRS), 3)) if vec_mode else np.zeros(len(BONE_PAIRS))
        if self.use_bone:
            Rwh = wrist_frame(hloc, self.mcp_kp) if self.bone_mode == "wrist" else None
            for pi, (a1, b1, a2, b2) in enumerate(BONE_PAIRS):
                d1 = hloc[b1] - hloc[a1]; d1 = d1 / (np.linalg.norm(d1) + 1e-9)
                d2 = hloc[b2] - hloc[a2]; d2 = d2 / (np.linalg.norm(d2) + 1e-9)
                if self.bone_mode == "wrist": bone_src[pi] = Rwh.T @ (d1 - d2)
                elif self.bone_mode == "direction": bone_src[pi] = d1 - d2
                else: bone_src[pi] = float(d1 @ d2)
        dir_src = np.zeros((5, 3))
        if self.w_dir > 0:
            for fi in range(5):
                v = hloc[self.tip_kp[fi]] - hloc[self.mcp_kp[fi]]
                dir_src[fi] = v / (np.linalg.norm(v) + 1e-9)
        return Lk, lap_src, lapO, bone_src, dir_src

    def _solve_manifold_frame(self, t, Q_ref, seed_full, obj_quat, obj_pos, inputs):
        """Re-solve manifold frame t warm-started from seed_full (smoothness toward the seed). Returns qfull."""
        Lk, lap_src, lapO, bone_src, dir_src = inputs
        self._cur_t = t
        self.data.qpos[:] = Q_ref[t]; mujoco.mj_forward(self.model, self.data)
        self._R_ref = self.data.xmat[self._base_body].reshape(3, 3).copy()
        self._trans_ref = Q_ref[t, self._trans_qadr].copy()
        qfull = Q_ref[t].copy()
        qfull[self._trans_qadr] = seed_full[self._trans_qadr]
        qfull[self._ball_qadr:self._ball_qadr + 4] = seed_full[self._ball_qadr:self._ball_qadr + 4]
        qfull[self._fin_qadr] = seed_full[self._fin_qadr]
        self.data.qpos[:] = seed_full; mujoco.mj_forward(self.model, self.data)
        self._R_prev = self.data.xmat[self._base_body].reshape(3, 3).copy()
        self._lin_prev = seed_full[self._lin_qadr].copy()
        phases = ([True] * self.warmup_iters if self.use_warmup else []) + [False] * self.sqp_iters
        for wu in phases:
            dq = self.solve_single_iteration(qfull, None, Lk, lap_src, lapO, obj_quat[t], obj_pos[t],
                                             bone_src, None, None, warmup=wu, dir_src=dir_src)
            if dq is None: break
            qvel = np.zeros(self.model.nv); qvel[self.dadr] = dq
            mujoco.mj_integratePos(self.model, qfull, qvel, 1.0)
            if (not wu) and self.sqp_tol > 0 and np.linalg.norm(dq) < self.sqp_tol: break
        return qfull

    def _repair_continuity(self, Q, Q_ref, hkp, obj_quat, obj_pos, contact_tgt, contact_mask, n,
                           pos_thr=0.05, tgt_thr=0.03):
        """CLASS-C fallback: detect isolated solved-wrist spikes (large jump to BOTH neighbors while the
        target keypoints are smooth), re-solve the frame seeded from t-1 AND t+1, and keep the candidate with
        the lowest continuity cost. No reference edit, no interpolation."""
        T = len(Q)
        def wp(q):
            self.data.qpos[:] = q; mujoco.mj_forward(self.model, self.data)
            return self.data.xpos[self._base_body].copy(), self.data.xmat[self._base_body].reshape(3, 3).copy()
        def gr(A, B):
            return np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))
        pw = [wp(Q[t]) for t in range(T)]; pos = [p for p, _ in pw]; Rw = [R for _, R in pw]
        repaired = []
        for t in range(1, T - 1):
            jp = np.linalg.norm(pos[t] - pos[t - 1]); jn = np.linalg.norm(pos[t] - pos[t + 1])
            gap = np.linalg.norm(pos[t - 1] - pos[t + 1]); tj = np.linalg.norm(hkp[t] - hkp[t - 1])
            if jp > pos_thr and jn > pos_thr and gap < 0.5 * max(jp, jn) and tj < tgt_thr:   # isolated spike, target smooth
                inp = self._frame_inputs(t, hkp, obj_quat, obj_pos, n, contact_tgt, contact_mask)
                cands = [Q[t].copy()]
                for seed in (Q[t - 1], Q[t + 1]):
                    cands.append(self._solve_manifold_frame(t, Q_ref, seed.copy(), obj_quat, obj_pos, inp))
                def cost(q):
                    p, R = wp(q)
                    return (np.linalg.norm(p - pos[t - 1]) + np.linalg.norm(p - pos[t + 1])
                            + 0.1 * (gr(R, Rw[t - 1]) + gr(R, Rw[t + 1])))
                best = min(cands, key=cost); Q[t] = best; pos[t], Rw[t] = wp(best); repaired.append(t)
        if repaired:
            print(f"    [continuity] re-solved {len(repaired)} isolated glitch frame(s): {repaired}", flush=True)
        return repaired

    def retarget_fingers(self, Q, hkp, obj_quat, obj_pos, contact_tgt=None, contact_mask=None,
                         anchor=0, warm_init=None, obj_quat_tgt=None, obj_pos_tgt=None):
        """Solve wrist+finger DOF for the whole trajectory, warm-started per frame.

        With `anchor` > 0 the solve starts there and propagates OUTWARD (anchor->T-1, then anchor-1->0,
        warm-started from the anchor), so the most-constrained grasp frame sets the basin. (Sequential
        warm-started solve = REGRIND App. A.2; outward order is ours.)

        Per frame: express the human keypoints in the object frame, (re)build the shared Delaunay graph +
        Laplacian over [hand keypoints + object points], compute the source bone directions, then iterate
        solve_single_iteration.

        Args:
            Q (np.ndarray): (T, nq) full qpos; body/object frozen, wrist+finger columns overwritten.
            hkp (np.ndarray): (T, 21, 3) human hand keypoints, world frame.
            obj_quat (np.ndarray): (T, 4) object orientation (wxyz).
            obj_pos (np.ndarray): (T, 3) object translation.
            contact_tgt (np.ndarray | None): (T, 5, 3) OBJECT-LOCAL surface targets per fingertip
                (thumb..pinky, from contact_target.fingertip_contacts); used when w_contact > 0.
            contact_mask (np.ndarray | None): (T, 5) bool, which fingertips the HUMAN has in contact.
        """
        T = Q.shape[0]
        n = self.nk + self.O.shape[0]
        # SOURCE/TARGET SEPARATION: source graph = [human, self.O(source)] @ obj_quat/obj_pos(source);
        # robot graph = [Wuji, Ot(target)] @ qt/pt(target). Same connectivity L (from source), fixed target object.
        qt = obj_quat if obj_quat_tgt is None else obj_quat_tgt
        pt = obj_pos if obj_pos_tgt is None else obj_pos_tgt
        Ot = self.O if self.O_target is None else self.O_target
        anchor = int(np.clip(anchor, 0, T - 1))
        # outward solve order: anchor -> end, then anchor-1 -> 0 (anchor=0 = the plain forward pass)
        order = list(range(anchor, T)) + list(range(anchor - 1, -1, -1))
        q_move = None if self.base_manifold else Q[anchor, self.qadr].copy()
        out = None if self.base_manifold else np.zeros((T, self.nm))
        q_prev = None                                                                 # previous frame's solved angles
        self._R_prev = None; self._lin_prev = None                                    # manifold smoothness refs
        Q_ref = Q.copy() if self.base_manifold else None                              # IMMUTABLE per-frame Kabsch base + object
        prev_full = None; anchor_full = None                                          # warm-start source (prev solved qpos)
        for t in tqdm(order, desc="  topo", leave=False):
            self._cur_t = t                                                           # for the distance-gated rotation prior
            if warm_init is not None and not self.base_manifold:                      # DIAGNOSTIC cross-init: start
                q_move = warm_init[t].copy()                                          # each frame from a supplied pose
            if anchor > 0 and t == anchor - 1 and not self.base_manifold:             # backward pass begins:
                q_move = out[anchor].copy()                                           # restart from the anchor's
                q_prev = out[anchor].copy()                                           # solution, not the last frame
            self._contact = (contact_tgt[t], contact_mask[t]) if contact_tgt is not None else None
            hloc = transform_points_world_to_local(obj_quat[t], obj_pos[t], hkp[t])   # human kp, object-local
            src = np.vstack([hloc, self.O])
            _, tets = create_interaction_mesh(src)                                    # shared Delaunay graph
            adj = get_adjacency_list(tets, n)
            L = exp_laplacian_matrix(src, adj, kappa=self.kappa, uniform=not self.exp_weight)
            lap_src = L @ src
            Lk, lapO = L[:, :self.nk], L[:, self.nk:] @ Ot        # robot object contribution uses the TARGET pts
            # source relative bone directions (cos bend) per adjacent pair, from the human hand. Only when
            # E_bone is on -- BONE_PAIRS indexes 4 kp/finger, so it's invalid for reduced-keypoint modes.
            vec_mode = self.bone_mode in ("direction", "wrist")                       # vector target vs cosine scalar
            bone_src = np.zeros((len(BONE_PAIRS), 3)) if vec_mode else np.zeros(len(BONE_PAIRS))
            if self.use_bone:
                Rwh = wrist_frame(hloc, self.mcp_kp) if self.bone_mode == "wrist" else None  # human wrist frame
                for pi, (a1, b1, a2, b2) in enumerate(BONE_PAIRS):
                    d1 = hloc[b1] - hloc[a1]; d1 = d1 / (np.linalg.norm(d1) + 1e-9)
                    d2 = hloc[b2] - hloc[a2]; d2 = d2 / (np.linalg.norm(d2) + 1e-9)
                    if self.bone_mode == "wrist":
                        bone_src[pi] = Rwh.T @ (d1 - d2)                                  # human dir-diff in HUMAN wrist frame
                    elif self.bone_mode == "direction":
                        bone_src[pi] = d1 - d2
                    else:
                        bone_src[pi] = float(d1 @ d2)
            dir_src = np.zeros((5, 3))                                                 # human MCP->TIP unit dir per finger
            if self.w_dir > 0:
                for fi in range(5):
                    v = hloc[self.tip_kp[fi]] - hloc[self.mcp_kp[fi]]
                    dir_src[fi] = v / (np.linalg.norm(v) + 1e-9)
            qfull = Q[t].copy()

            if self.base_manifold:                                                    # --- S3 manifold: integrate on SO(3) ---
                # IMMUTABLE reference for THIS frame = the per-frame Kabsch base + current object pose (Q_ref[t])
                self.data.qpos[:] = Q_ref[t]; mujoco.mj_forward(self.model, self.data)
                self._R_ref = self.data.xmat[self._base_body].reshape(3, 3).copy()    # Kabsch reference orientation
                self._trans_ref = Q_ref[t, self._trans_qadr].copy()
                qfull = Q_ref[t].copy()                                                # start from ref (object pose + Kabsch base)
                if anchor > 0 and t == anchor - 1 and anchor_full is not None:         # backward pass: reset warm-start to anchor
                    prev_full = anchor_full
                    self.data.qpos[:] = anchor_full; mujoco.mj_forward(self.model, self.data)
                    self._R_prev = self.data.xmat[self._base_body].reshape(3, 3).copy()
                    self._lin_prev = anchor_full[self._lin_qadr].copy()
                if prev_full is not None:                                              # WARM-START the hand from the prev solution
                    qfull[self._trans_qadr] = prev_full[self._trans_qadr]             #   (object pose stays this frame's)
                    qfull[self._ball_qadr:self._ball_qadr + 4] = prev_full[self._ball_qadr:self._ball_qadr + 4]
                    qfull[self._fin_qadr] = prev_full[self._fin_qadr]
                # warmup_from_frame0 (opt-in): let the paper E_bone warmup run even at f0 (no _R_prev). With _R_prev
                # None the base-smoothness term is inactive, so warmup is finger-only there. warmup_freeze_base (opt-in)
                # additionally zeroes the base dq during ALL warmup iters => the wrist/root is provably fixed while only
                # the fingers articulate. Both default off => production/A behaviour is byte-identical.
                do_warm = self.use_warmup and (self._R_prev is not None or getattr(self, "warmup_from_frame0", False))
                phases = ([True] * self.warmup_iters if do_warm else []) + [False] * self.sqp_iters
                nwarm = self.warmup_iters if do_warm else 0
                _wl = getattr(self, "_warm_log", None)
                if _wl is not None:
                    _b0 = np.concatenate([qfull[self._trans_qadr], qfull[self._ball_qadr:self._ball_qadr + 4]]).copy()
                    _f0 = qfull[self._fin_qadr].copy(); _ran = 0
                # gated SOLVER-QUALITY SAFEGUARD (sqp_guard, default off => byte-identical). Guarded SQP: keep the best
                # feasible iterate; reject an optimal_inaccurate step or any step that WORSENS the true nonlinear merit
                # (E_IM residual + base-trans prior + pen-slack violation); never overwrite the last valid iterate; early
                # stop after a stable small accepted step. _trace_t records the per-iteration trace of one frame.
                _guard = getattr(self, "sqp_guard", False); _trace = (getattr(self, "_trace_t", None) == t)
                if _guard or _trace:
                    _lam = float(self.lambda_im_sched[t]) if self.lambda_im_sched is not None else self.lambda_im
                    _ims = _lam / (lap_src.shape[0] if self.im_normalize else 1.0)
                    def _merit(_qf):
                        self.data.qpos[:] = _qf; mujoco.mj_forward(self.model, self.data)
                        _pk = np.array([self.data.xpos[b] for b in self.kb])
                        for _fi in range(1, 6):
                            _k = self.tip_kp[_fi - 1]; _b4 = self.kb[_k]; _loc = self.tip_local.get(_k)
                            if _loc is not None: _pk[_k] = self.data.xpos[_b4] + self.data.xmat[_b4].reshape(3, 3) @ _loc
                        _pl = transform_points_world_to_local(qt[t], pt[t], _pk)
                        _m = _ims * float(np.sum((Lk @ _pl + lapO - lap_src) ** 2))            # true nonlinear E_IM residual
                        if self.base_reg_trans > 0 and self._trans_ref is not None:
                            _m += self.base_reg_trans * float(np.sum((_qf[self._trans_qadr] - self._trans_ref) ** 2))
                        return _m + (self.w_s / 2.0) * float(getattr(self, "_pen_slack_max", 0.0)) ** 2  # + pen violation
                    if _trace and not hasattr(self, "_iter_trace"): self._iter_trace = []
                    _conv_q = None; _conv_m = _last_m = _merit(qfull)                 # last CONVERGED iterate (||dq||~0) + merit
                for _i, wu in enumerate(phases):
                    dq = self.solve_single_iteration(qfull, None, Lk, lap_src, lapO,
                                                     qt[t], pt[t], bone_src, None, None,
                                                     warmup=wu, dir_src=dir_src)
                    if dq is None:
                        break
                    if wu and getattr(self, "warmup_freeze_base", False):
                        dq[:self.n_wrist] = 0.0                                       # finger-only warmup: freeze wrist/root
                    qvel = np.zeros(self.model.nv); qvel[self.dadr] = dq
                    if _guard and not wu:                                            # apply EVERY step (identical trajectory to
                        _qt_ = qfull.copy(); mujoco.mj_integratePos(self.model, _qt_, qvel, 1.0)  #   the unguarded solve; do NOT
                        _st = getattr(self, "_last_status", "optimal"); _mt = _merit(_qt_); _nd = float(np.linalg.norm(dq))  # break on
                        if _trace: self._iter_trace.append(dict(t=int(t), it=int(_i), status=_st, ndq=_nd, merit=_mt))  # inaccurate --
                        qfull = _qt_; _last_m = _mt                                  #   breaking early diverges from frozen). Track
                        if _nd < getattr(self, "_guard_tol", 1e-4) or _st != "optimal":  # the CONVERGED (or last non-inaccurate)
                            _conv_q = _qt_.copy(); _conv_m = _mt                     #   iterate; the f127 flick is a LATER step that
                        # NB: NO early-stop -- the reset settling needs the full iteration budget to descend into the
                        # under-lip basin (an early ||dq||~0 plateau sits ON TOP; stopping there traps the thumb).
                    else:
                        mujoco.mj_integratePos(self.model, qfull, qvel, 1.0)         # quaternion via exp-map, slides/fingers add
                        if _trace and not wu:
                            self._iter_trace.append(dict(t=int(t), it=int(_i), status=getattr(self, "_last_status", "?"), ndq=float(np.linalg.norm(dq)), merit=_merit(qfull)))
                    if _wl is not None and wu:
                        _ran += 1
                    if _wl is not None and (_i + 1) == nwarm:                          # snapshot right after the warmup phase
                        _b1 = np.concatenate([qfull[self._trans_qadr], qfull[self._ball_qadr:self._ball_qadr + 4]])
                        _wl.append(dict(t=int(t), ran=_ran, dfin=float(np.linalg.norm(qfull[self._fin_qadr] - _f0)), dbase=float(np.linalg.norm(_b1 - _b0))))
                    if (not wu) and self.sqp_tol > 0 and np.linalg.norm(dq) < self.sqp_tol:
                        break
                if _guard and _conv_q is not None:                                   # revert ONLY on a genuine POST-CONVERGENCE
                    self.data.qpos[:] = qfull; mujoco.mj_forward(self.model, self.data); _wf = self.data.xpos[self._base_body].copy()
                    self.data.qpos[:] = _conv_q; mujoco.mj_forward(self.model, self.data); _wc = self.data.xpos[self._base_body].copy()
                    if float(np.linalg.norm(_wf - _wc)) > getattr(self, "_guard_wjump", 0.10) and _last_m > _conv_m + 1e-9:
                        qfull = _conv_q                                              #   spike: the SQP had converged (small ||dq||),
                                                                                     #   then a later step teleported the WORLD wrist
                                                                                     #   >10cm AND worsened the merit -> restore the
                                                                                     #   converged iterate. Smooth frames untouched.
                if _wl is not None and nwarm == 0:
                    _wl.append(dict(t=int(t), ran=0, dfin=0.0, dbase=0.0))
                Q[t] = qfull                                                          # object DoF preserved (integrate touched hand only)
                if t == anchor:
                    anchor_full = qfull.copy()
                prev_full = qfull.copy()                                              # next frame warm-starts from here
                self.data.qpos[:] = qfull; mujoco.mj_forward(self.model, self.data)
                self._R_prev = self.data.xmat[self._base_body].reshape(3, 3).copy()   # for next frame's geodesic smoothness
                self._lin_prev = qfull[self._lin_qadr].copy()
                continue

            q_ref = Q[t, self.qadr].copy()                                            # (wrist regularizer target if enabled)
            # NB: do NOT reset the wrist to Stage A each frame -- let it warm-start across frames and
            # roll freely (like the original free-wrist B0), else the hand never reaches the object.
            if self.use_warmup and q_prev is not None:                                # warmup: bone+smooth init (Eq.2)
                for _ in range(self.warmup_iters):
                    dq = self.solve_single_iteration(qfull, q_move, Lk, lap_src, lapO,
                                                     qt[t], pt[t], bone_src, q_ref, q_prev,
                                                     warmup=True, dir_src=dir_src)
                    if dq is None:
                        break
                    q_move = q_move + dq
            for _ in range(self.sqp_iters):                                           # main solve (sqp_iters = MAX budget)
                dq = self.solve_single_iteration(qfull, q_move, Lk, lap_src, lapO,
                                                 qt[t], pt[t], bone_src, q_ref, q_prev,
                                                 warmup=False, dir_src=dir_src)
                if dq is None:
                    break
                q_move = q_move + dq
                if self.sqp_tol > 0 and np.linalg.norm(dq) < self.sqp_tol:            # converged (paper: iterate to tol)
                    break
            out[t] = q_move
            q_prev = q_move.copy()
        if self.base_manifold and self.continuity_fallback:                       # CLASS-C rerun-and-select fallback
            self._repair_continuity(Q, Q_ref, hkp, obj_quat, obj_pos, contact_tgt, contact_mask, n)
        if not self.base_manifold:
            for k, a in enumerate(self.qadr):
                Q[:, a] = out[:, k]


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    stageA_npz, out_npz, model_xml, object_obj, handkp_npy = args[:5]
    n_obj = int(sys.argv[sys.argv.index("--pts") + 1]) if "--pts" in sys.argv else 100
    kappa = float(sys.argv[sys.argv.index("--kappa") + 1]) if "--kappa" in sys.argv else 30.0
    w_bone = float(sys.argv[sys.argv.index("--w-bone") + 1]) if "--w-bone" in sys.argv else 0.1
    w_smooth = float(sys.argv[sys.argv.index("--w-smooth") + 1]) if "--w-smooth" in sys.argv else 0.05
    exp_weight = "--exp-weight" in sys.argv                          # else uniform (OmniRetarget-style)
    use_bone = "--bone" in sys.argv
    use_smooth = "--smooth" in sys.argv                             # E_reg temporal smoothness (glitch fix)
    use_warmup = "--warmup" in sys.argv                            # TopoRetarget per-frame E_bone warmup init
    soft_pen = "--soft-pen" in sys.argv                             # slack object non-penetration (Track B)
    obj_name = os.path.splitext(os.path.basename(object_obj))[0]     # object geom name (e.g. "table")

    Mf = mujoco.MjModel.from_xml_path(model_xml)
    Mb = mujoco.MjModel.from_xml_path(model_xml.replace("_wuji_w_", "_wuji_welded_w_").replace("_coacd", ""))
    npz = np.load(stageA_npz, allow_pickle=True)
    qb = npz["qpos"]
    T = qb.shape[0]
    hkp = np.load(handkp_npy)                                        # (T, 2, 21, 3): [left, right]

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
    obj_quat = Q[:, Mf.nq - 4:Mf.nq]
    O, _ = load_object_data(object_obj, smpl_scale=1.0, sample_count=n_obj)
    print(f"[topo] {T} frames | {O.shape[0]} object points | 21 kp/hand | wrist+fingers"
          f" | Laplacian={'exp(kappa=%g)' % kappa if exp_weight else 'uniform'}"
          f" | bone={'ON (w=%g)' % w_bone if use_bone else 'off'}"
          f" | smooth={'ON (w=%g)' % w_smooth if use_smooth else 'off'} | warmup={'ON' if use_warmup else 'off'}"
          f" | soft-pen={'ON' if soft_pen else 'off'}")
    for si, side in enumerate(("left", "right")):
        print(f"  solving {side} hand ...")
        TopoRetargeter(Mf, side, O, kappa=kappa, exp_weight=exp_weight, use_bone=use_bone, w_bone=w_bone,
                       use_smooth=use_smooth, w_smooth=w_smooth, use_warmup=use_warmup,
                       soft_pen=soft_pen, obj_name=obj_name).retarget_fingers(Q, hkp[:, si], obj_quat, obj_pos)

    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    extra = {"human_joints": npz["human_joints"]} if "human_joints" in npz.files else {}
    np.savez(out_npz, qpos=Q, fps=30, **extra)
    print(f"  wrote {out_npz}  qpos {Q.shape}")


if __name__ == "__main__":
    main()
