"""V2: Omni body -> quick anatomical finger fit -> coupled arms/fingers -> direct export.

The joint refinement uses native Clarabel matrices, without CVXPY compilation.
HUMOTO supplies Mixamo landmarks, not MANO coefficients. The initializer accepts
21 anatomical landmarks in wrist, thumb/index/middle/ring/pinky MCP/PIP/DIP/TIP
order; MANO-derived landmarks must first be converted to this explicit order.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

import clarabel
import mujoco
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG.parent))
from holosoma_retargeting.wuji.topoRetarget import (
    exp_laplacian_matrix, wrist_frame, create_interaction_mesh, get_adjacency_list,
)
from holosoma_retargeting.retarget_pipeline.clarabel_step import make_settings, solve_clarabel_qp_step

SIDES = ('left', 'right')
PREFIX = ('wj_left', 'wjr_right')
ARM = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
       'wrist_roll', 'wrist_pitch', 'wrist_yaw')
MCP = [1, 5, 9, 13, 17]
TIPS = [4, 8, 12, 16, 20]
SEGMENTS = [(1+4*f+i, 2+4*f+i) for f in range(5) for i in range(3)]


@dataclass
class Settings:
    version: str = '2.0-pilot'
    fps: float = 30.
    init_iterations: int = 6
    max_iterations: int = 15
    first_frame_iterations: int = 40
    min_iterations: int = 3
    convergence_step: float = 2e-4
    trust_radius: float = .45
    collision_detect: float = .025
    collision_tolerance: float = .001
    collision_weight: float = 1e5
    lambda_im: float = 500.
    tip_weight: float = 10.
    bone_weight: float = .10
    fingertip_weight: float = 25.
    contact_weight: float = 100.
    palm_axis_weight: float = .5
    temporal_weight: float = .15
    finger_prior_weight: float = .005
    arm_prior_contact: float = .002
    arm_prior_no_contact: float = .05
    joint_rate_limit: float = .20  # rad/frame at 30 Hz
    damping: float = .001


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def joint_map(m):
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j):
            (int(m.jnt_qposadr[j]), 7 if m.jnt_type[j] == 0 else 4 if m.jnt_type[j] == 1 else 1)
            for j in range(m.njnt)}


def structurally_adjacent(m, a, b):
    for child, ancestor in ((a, b), (b, a)):
        count = 0
        while child > 0 and count <= 2:
            if child == ancestor:
                return True
            count += int(m.body_jntnum[child])
            child = int(m.body_parentid[child])
    return False


def normalized(v):
    return v / max(float(np.linalg.norm(v)), 1e-8)


class CoupledRefiner:
    def __init__(self, model, object_name, settings=None):
        self.m = model
        self.d = mujoco.MjData(model)
        self.cfg = settings or Settings()
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
                 for j in range(model.njnt)]
        self.joints = []
        self.side_columns = []
        self.finger_columns = []
        self.arm_columns = []
        for side, prefix in zip(SIDES, PREFIX):
            arms = [names.index(f'{side}_{a}_joint') for a in ARM]
            fingers = [j for j, n in enumerate(names) if n.startswith(prefix) and 'finger' in n]
            assert len(fingers) == 20
            offset = len(self.joints)
            self.arm_columns.append(np.arange(offset, offset+7))
            self.finger_columns.append(np.arange(offset+7, offset+27))
            self.side_columns.append(np.arange(offset, offset+27))
            self.joints.extend(arms+fingers)
        self.qadr = model.jnt_qposadr[self.joints].astype(int)
        self.dadr = model.jnt_dofadr[self.joints].astype(int)
        self.lo = model.jnt_range[self.joints, 0].copy()
        self.hi = model.jnt_range[self.joints, 1].copy()
        self.n = len(self.joints)
        assert self.n == 54
        self.kb, self.tipgeom = [], []
        for prefix in PREFIX:
            bn = [prefix+'_palm_link']+[f'{prefix}_finger{f}_link{l}'
                  for f in range(1, 6) for l in (1, 3, 4, 4)]
            bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in bn]
            assert min(bids) >= 0
            self.kb.append(bids)
            tips = []
            for f in range(1, 6):
                bid = bids[4*f]
                found = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid
                         and model.geom_dataid[g] >= 0 and
                         f'finger{f}_tip_link' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH,
                                                  int(model.geom_dataid[g])) or '')
                         and (model.geom_contype[g] or model.geom_conaffinity[g])]
                assert len(found) == 1
                tips.append(found[0])
            self.tipgeom.append(tips)
        self.kb = np.array(self.kb)
        self.tipgeom = np.array(self.tipgeom)
        moving_body = np.zeros(model.nbody, bool)
        selected = set(self.joints)
        for b in range(1, model.nbody):
            own = range(int(model.body_jntadr[b]), int(model.body_jntadr[b]+model.body_jntnum[b]))
            moving_body[b] = moving_body[model.body_parentid[b]] or any(j in selected for j in own)
        active = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
        objects = [g for g in active if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '').startswith(object_name+'_piece')]
        planes = [g for g in active if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
        assert objects and planes, 'V2 requires active object convexes and a ground plane'
        robot = [g for g in active if g not in set(objects+planes)]
        moving = [g for g in robot if moving_body[model.geom_bodyid[g]]]
        pairs = []
        for a in moving:
            pairs.extend((a, b, 0) for b in objects)
            pairs.extend((a, b, 1) for b in planes)
        for ia, a in enumerate(robot):
            for b in robot[ia+1:]:
                ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
                if not (moving_body[ba] or moving_body[bb]):
                    continue
                if model.body_weldid[ba] == model.body_weldid[bb] or structurally_adjacent(model, ba, bb):
                    continue
                if not ((model.geom_contype[a] & model.geom_conaffinity[b]) or
                        (model.geom_contype[b] & model.geom_conaffinity[a])):
                    continue
                # Respect explicit MJCF contact exclusions as well as the common structural filter.
                signature = (min(ba, bb) << 16) + max(ba, bb)
                if signature in model.exclude_signature:
                    continue
                pairs.append((a, b, 2))
        self.pairs = np.asarray(pairs, int)
        self.is_plane = model.geom_type[self.pairs[:, 1]] == mujoco.mjtGeom.mjGEOM_PLANE
        self.pair_categories = {k: int(np.sum(self.pairs[:, 2] == i))
                                for i, k in enumerate(('object', 'ground', 'self'))}
        self.solver_settings = make_settings(max_iter=100)

    def fk(self, q):
        self.d.qpos[:] = q
        # Position and Jacobian computations need kinematics and center-of-mass transforms.
        # Collision distances are explicitly queried below; no dynamics solve is needed.
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def keypoints(self, with_jac=True):
        points = self.d.xpos[self.kb].copy()
        for side in range(2):
            points[side, TIPS] = self.d.geom_xpos[self.tipgeom[side]]
        if not with_jac:
            return points
        jac = np.zeros((2, 21, 3, self.n))
        for side in range(2):
            for i, b in enumerate(self.kb[side]):
                jp = np.zeros((3, self.m.nv))
                mujoco.mj_jac(self.m, self.d, jp, None, points[side, i], int(b))
                jac[side, i] = jp[:, self.dadr]
        return points, jac

    def quick_fingers(self, Q, source):
        """Short bone-direction IK in semantic palm coordinates; no object or root solve."""
        out = Q.copy()
        targets = []
        for h in source:
            target = []
            for side in range(2):
                R = wrist_frame(h[side], MCP)
                target.append(np.array([R.T @ normalized(h[side, b]-h[side, a]) for a, b in SEGMENTS]))
            targets.append(target)
        for t in range(len(out)):
            if t:
                for cols in self.finger_columns:
                    out[t, self.qadr[cols]] = out[t-1, self.qadr[cols]]
            for _ in range(self.cfg.init_iterations):
                self.fk(out[t])
                points, jac = self.keypoints()
                for side, cols in enumerate(self.finger_columns):
                    R = wrist_frame(points[side], MCP)
                    residual, J = [], []
                    for i, (a, b) in enumerate(SEGMENTS):
                        vec = points[side, b]-points[side, a]
                        length = max(np.linalg.norm(vec), 1e-6)
                        direction = vec/length
                        deriv = (np.eye(3)-np.outer(direction, direction))/length @ (jac[side, b]-jac[side, a])
                        residual.extend(R.T @ direction - targets[t][side][i])
                        J.extend((R.T @ deriv)[:, cols])
                    J, residual = np.asarray(J), np.asarray(residual)
                    dq = np.linalg.solve(J.T@J + .03*np.eye(20), -J.T@residual)
                    dq *= min(1., .6/max(np.linalg.norm(dq), 1e-9))
                    adr = self.qadr[cols]
                    out[t, adr] = np.clip(out[t, adr]+dq, self.lo[cols], self.hi[cols])
        return out

    def collisions(self, q, derivatives=True):
        self.fk(q)
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        lower = np.linalg.norm(self.d.geom_xpos[a]-self.d.geom_xpos[b], axis=1)-self.m.geom_rbound[a]-self.m.geom_rbound[b]
        candidates = np.flatnonzero((lower < self.cfg.collision_detect) | self.is_plane)
        distances, gradients, pair_ids = [], [], []
        ft = np.zeros(6)
        for i in candidates:
            ga, gb, _ = self.pairs[i]
            dist = float(mujoco.mj_geomDistance(self.m, self.d, int(ga), int(gb), self.cfg.collision_detect, ft))
            if dist >= self.cfg.collision_detect:
                continue
            delta = ft[:3]-ft[3:]
            norm = np.linalg.norm(delta)
            if norm < 1e-10:
                continue
            distances.append(dist)
            pair_ids.append(i)
            if derivatives:
                ja, jb = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
                mujoco.mj_jac(self.m, self.d, ja, None, ft[:3], int(self.m.geom_bodyid[ga]))
                mujoco.mj_jac(self.m, self.d, jb, None, ft[3:], int(self.m.geom_bodyid[gb]))
                gradients.append(np.sign(dist)*(delta/norm) @ (ja-jb)[:, self.dadr])
        return np.asarray(distances), np.asarray(gradients).reshape(-1, self.n), np.asarray(pair_ids, int)

    def frame_target(self, human, O, object_pose, contact_mask, surface_targets):
        R = Rotation.from_quat(object_pose[[4, 5, 6, 3]]).as_matrix()
        graphs, directions, palm = [], [], []
        for side in range(2):
            local = (human[side]-object_pose[:3]) @ R
            vertices = np.vstack([local, O])
            _, tets = create_interaction_mesh(vertices)
            L = exp_laplacian_matrix(vertices, get_adjacency_list(tets, len(vertices)), kappa=30.)
            graphs.append((L[:, :21], L @ vertices, L[:, 21:] @ O))
            Rh = wrist_frame(human[side], MCP)
            directions.append(np.array([Rh.T @ normalized(human[side, b]-human[side, a]) for a, b in SEGMENTS]))
            palm.append(Rh)
        return dict(human=human, O=O, object_pose=object_pose, R=R, graphs=graphs,
                    directions=directions, palm=palm, contact=contact_mask,
                    surface=surface_targets)

    def residuals(self, q, target, q_body, q_finger, previous):
        self.fk(q)
        points, jac = self.keypoints()
        residuals, matrices = [], []
        def add(r, J, weight):
            residuals.append(np.asarray(r).reshape(-1)*np.sqrt(weight))
            matrices.append(np.asarray(J).reshape(-1, self.n)*np.sqrt(weight))
        for side in range(2):
            Lk, desired, object_part = target['graphs'][side]
            local = (points[side]-target['object_pose'][:3]) @ target['R']
            Jlocal = np.einsum('ij,kjn->kin', target['R'].T, jac[side])
            lap = Lk @ local + object_part
            Jlap = np.einsum('ik,kjn->ijn', Lk, Jlocal)
            amp = np.ones((len(lap), 1));amp[TIPS] = np.sqrt(self.cfg.tip_weight)
            add((lap-desired)*amp, Jlap*amp[:, :, None], self.cfg.lambda_im/len(lap))
            add(points[side, TIPS]-target['human'][side, TIPS], jac[side, TIPS], self.cfg.fingertip_weight)
            active = target['contact'][side]
            if active.any():
                # Fixed source scene surface targets, never targets from an optimized floating hand.
                add(points[side, np.array(TIPS)[active]]-target['surface'][side, active],
                    jac[side, np.array(TIPS)[active]], self.cfg.contact_weight)
            Rw = wrist_frame(points[side], MCP)
            for i, (a, b) in enumerate(SEGMENTS):
                vec = points[side, b]-points[side, a];length = max(np.linalg.norm(vec), 1e-6)
                direction = vec/length
                J = (np.eye(3)-np.outer(direction, direction))/length @ (jac[side, b]-jac[side, a])
                J = Rw.T @ J
                # Local bone orientation is invariant to arm motion; palm MCP origins do not move with finger flexion.
                J[:, self.arm_columns[side]] = 0.
                add(Rw.T @ direction-target['directions'][side][i], J, self.cfg.bone_weight)
            # Align semantic palm axes through analytic angular Jacobian of the palm rigid body.
            jr = np.zeros((3, self.m.nv))
            mujoco.mj_jac(self.m, self.d, None, jr, points[side, 0], int(self.kb[side, 0]))
            for ax in (0, 2):
                v = Rw[:, ax]
                skew = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
                add(v-target['palm'][side][:, ax], -skew@jr[:, self.dadr], self.cfg.palm_axis_weight)
            arm = self.arm_columns[side]
            w = self.cfg.arm_prior_contact if active.any() else self.cfg.arm_prior_no_contact
            add(q[self.qadr[arm]]-q_body[self.qadr[arm]], np.eye(self.n)[arm], w)
            fingers = self.finger_columns[side]
            add(q[self.qadr[fingers]]-q_finger[self.qadr[fingers]], np.eye(self.n)[fingers], self.cfg.finger_prior_weight)
        if previous is not None:
            add(q[self.qadr]-previous[self.qadr], np.eye(self.n), self.cfg.temporal_weight)
        return np.concatenate(residuals), np.vstack(matrices)

    def subproblem(self, q, residual, J, dist, gradient, previous):
        n, k = self.n, len(dist)
        H = 2*(J.T@J+self.cfg.damping*np.eye(n))
        P = sparse.block_diag([sparse.csc_matrix(H), self.cfg.collision_weight*sparse.eye(k)], format='csc')
        linear = np.r_[2*J.T@residual, np.zeros(k)]
        low, high = self.lo-q[self.qadr], self.hi-q[self.qadr]
        if previous is not None:
            low = np.maximum(low, previous[self.qadr]-self.cfg.joint_rate_limit-q[self.qadr])
            high = np.minimum(high, previous[self.qadr]+self.cfg.joint_rate_limit-q[self.qadr])
        I, Z = sparse.eye(n, format='csc'), sparse.csc_matrix((n, k))
        blocks = [sparse.hstack([I, Z]), sparse.hstack([-I, Z])]
        rhs = [high, -low]
        if k:
            blocks += [sparse.hstack([-sparse.csc_matrix(gradient), -sparse.eye(k)]),
                       sparse.hstack([sparse.csc_matrix((k, n)), -sparse.eye(k)])]
            rhs += [dist+self.cfg.collision_tolerance, np.zeros(k)]
        A = sparse.vstack(blocks, format='csc')
        b = np.concatenate(rhs)
        cones = [clarabel.NonnegativeConeT(len(b))]
        soc = sparse.vstack([sparse.csc_matrix((1, n+k)), sparse.hstack([-I, Z])], format='csc')
        A = sparse.vstack([A, soc], format='csc')
        b = np.r_[b, self.cfg.trust_radius, np.zeros(n)]
        cones.append(clarabel.SecondOrderConeT(n+1))
        x, info = solve_clarabel_qp_step(P, linear, A, b, cones, self.solver_settings)
        info.update(slack_max_m=float(x[n:].max()) if k else 0., collision_rows=k)
        return x[:n], info

    def refine(self, body_Q, init_Q, human, O, pose, contact, surface, progress=None):
        Q = body_Q.copy()
        previous = None
        records = []
        for t in range(len(Q)):
            q = init_Q[t].copy()
            if previous is not None:
                q[self.qadr] = previous[self.qadr]
            target = self.frame_target(human[t], O, pose[t], contact[t], surface[t])
            limit = self.cfg.first_frame_iterations if t == 0 else self.cfg.max_iterations
            details, accepted, failed_linesearch = [], 0, False
            for it in range(limit):
                residual, J = self.residuals(q, target, body_Q[t], init_Q[t], previous)
                dist, G, _ = self.collisions(q)
                step, info = self.subproblem(q, residual, J, dist, G, previous)
                merit = float(residual@residual + .5*self.cfg.collision_weight*np.sum(np.maximum(-dist-self.cfg.collision_tolerance, 0.)**2))
                chosen = None
                for alpha in (1., .5, .25, .125):
                    trial = q.copy();trial[self.qadr] += alpha*step
                    r, _ = self.residuals(trial, target, body_Q[t], init_Q[t], previous)
                    dd, _, _ = self.collisions(trial, derivatives=False)
                    value = float(r@r + .5*self.cfg.collision_weight*np.sum(np.maximum(-dd-self.cfg.collision_tolerance, 0.)**2))
                    if value <= merit + 1e-9:
                        chosen = trial
                        break
                info.update(iteration=it, step_norm=float(np.linalg.norm(step)), merit=merit,
                            accepted=chosen is not None, alpha=alpha if chosen is not None else 0.)
                details.append(info)
                if chosen is None:
                    failed_linesearch = True
                    break
                q = chosen;accepted += 1
                if it+1 >= self.cfg.min_iterations and np.linalg.norm(alpha*step) < self.cfg.convergence_step:
                    break
            Q[t] = q
            previous = q.copy()
            records.append(dict(frame=t, iterations=len(details), accepted=accepted,
                                linesearch_stopped=failed_linesearch, steps=details))
            if progress and (t % 10 == 0 or t == len(Q)-1):
                progress(t+1, len(Q), records[-1])
        frozen = np.setdiff1d(np.arange(self.m.nq), self.qadr)
        assert np.array_equal(Q[:, frozen], body_Q[:, frozen])
        assert np.isfinite(Q).all()
        assert np.all(Q[:, self.qadr] >= self.lo-1e-7) and np.all(Q[:, self.qadr] <= self.hi+1e-7)
        return Q, records


def initial_body(full_model, body_model, body_q):
    jf, ja = joint_map(full_model), joint_map(body_model)
    Q = np.tile(full_model.qpos0, (len(body_q), 1))
    for name, (a, width) in jf.items():
        if name in ja:
            b, oldwidth = ja[name]
            assert width == oldwidth
            Q[:, a:a+width] = body_q[:, b:b+width]
    Q[:, :7] = body_q[:, :7]
    Q[:, -7:] = body_q[:, -7:]
    return Q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical', type=Path, required=True)
    parser.add_argument('--body', type=Path, required=True)
    parser.add_argument('--body-model', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--object', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, help='Development smoke test only; omitted for full sequences')
    parser.add_argument('--settings', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = Settings(**json.loads(args.settings.read_text())) if args.settings else Settings()
    start = time.perf_counter()
    z = np.load(args.canonical, allow_pickle=False)
    human, pose, O = z['hand_kp_scaled'], z['object_pose_mujoco'], z['object_points_scaled']
    body_q = np.load(args.body)['qpos']
    assert len(body_q) == len(human) == len(pose)
    if args.frames:
        human, pose, body_q = human[:args.frames], pose[:args.frames], body_q[:args.frames]
    m = mujoco.MjModel.from_xml_path(str(args.model))
    ma = mujoco.MjModel.from_xml_path(str(args.body_model))
    Qbody = initial_body(m, ma, body_q)
    del ma
    refiner = CoupledRefiner(m, args.object, settings)
    # Source labels/targets use exactly the evaluation's original scaled object mesh.
    import trimesh
    mesh = trimesh.load(PKG/'models'/args.object/f'{args.object}.obj', force='mesh')
    mesh.apply_scale(float(z['scale']))
    pq = trimesh.proximity.ProximityQuery(mesh)
    R = Rotation.from_quat(pose[:, [4, 5, 6, 3]]).as_matrix()
    tips = human[:, :, TIPS]
    local = np.einsum('tij,thki->thkj', R, tips-pose[:, None, None, :3])
    near, distances = [], []
    flat = local.reshape(-1, 3)
    for i in range(0, len(flat), 16):
        a, b, _ = pq.on_surface(flat[i:i+16]);near.append(a);distances.append(b)
    near = np.concatenate(near).reshape(local.shape)
    contact = np.concatenate(distances).reshape(local.shape[:-1]) < .03
    surface = np.einsum('tij,thkj->thki', R, near)+pose[:, None, None, :3]
    prep_seconds = time.perf_counter()-start
    ti = time.perf_counter();Qinit = refiner.quick_fingers(Qbody, human);init_seconds = time.perf_counter()-ti
    np.savez_compressed(args.output/'finger_initialization.npz', qpos=Qinit, fps=30)
    def progress(done, total, frame):
        write_json(args.output/'status.json', dict(stage='joint_refinement', completed_frames=done,
            total_frames=total, latest_iterations=frame['iterations'], elapsed_seconds=time.perf_counter()-start))
        print(f'V2 {done}/{total} frames, {time.perf_counter()-start:.1f}s', flush=True)
    tr = time.perf_counter()
    Q, records = refiner.refine(Qbody, Qinit, human, O, pose, contact, surface, progress)
    refine_seconds = time.perf_counter()-tr
    np.savez_compressed(args.output/'final_v2.npz', qpos=Q, fps=30, human_joints=z['human_joints_scaled'][:len(Q)])
    write_json(args.output/'iterations.json', records)
    result = dict(version=settings.version, frames=len(Q), settings=asdict(settings),
        preparation_seconds=prep_seconds, finger_initialization_seconds=init_seconds,
        joint_refinement_seconds=refine_seconds, total_refinement_process_seconds=time.perf_counter()-start,
        body_source=str(args.body), body_reused=True, model=str(args.model), canonical=str(args.canonical),
        active_dofs=54, frozen_coordinates_exact=True, final_uses_refined_fingers=True,
        collision_pair_counts=refiner.pair_categories,
        maximum_step_slack_m=max(s['slack_max_m'] for r in records for s in r['steps']),
        solver_failures=0, total_subproblems=sum(r['iterations'] for r in records),
        source_type='HUMOTO Mixamo anatomical landmarks; no MANO coefficients or SMPL-X fit',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(args.output/'run.json', result)
    write_json(args.output/'status.json', dict(stage='complete', frames=len(Q), elapsed_seconds=time.perf_counter()-start))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
