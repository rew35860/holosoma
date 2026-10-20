"""Read-only trajectory audit and exact single-frame replays of selected geometry variant.

Writes a separate diagnostics directory; never rewrites any trajectory or setting.
One CPU thread, no CUDA. Frame indices are zero-based at 30 Hz.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import csv
import json
import time
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import OUT, COHORT, destination, sha
from solver import load, target_at, mujoco
import pipeline_v2 as core

DEST = OUT / 'diagnostics/jitter_3s'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def rot(q):
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()


def angle(a, b):
    return float(np.degrees(Rotation.from_matrix(a.T @ b).magnitude()))


def pair_info(ref, index):
    ga, gb, cat = map(int, ref.pairs[index])
    def info(g):
        m = ref.m
        mid = int(m.geom_dataid[g])
        return dict(id=g, name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g),
                    body=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])),
                    mesh=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid) if mid >= 0 else None)
    return dict(pair_index=int(index), category=('object', 'ground', 'self')[cat], a=info(ga), b=info(gb))


def coll_summary(ref, dist, ids):
    result = {}
    for cat, name in enumerate(('object', 'ground', 'self')):
        rows = np.flatnonzero(ref.pairs[ids, 2] == cat)
        if not len(rows):
            result[name] = dict(depth_mm=0., pair=None)
            continue
        i = rows[np.argmin(dist[rows])]
        result[name] = dict(depth_mm=max(0., -float(dist[i])) * 1000,
                            distance_mm=float(dist[i]) * 1000, pair=pair_info(ref, ids[i]))
    return result


def cost_parts(ref, r, target, dist):
    result = {}; offset = 0
    for side in range(2):
        sizes = [('interaction_mesh', len(target['graphs'][side][0]) * 3),
                 ('fingertip_tracking', 15), ('source_point_attraction', int(target['contact'][side].sum()) * 3),
                 ('bone', 45), ('palm', 6), ('omni_arm_prior', 7), ('finger_init_prior', 20)]
        for name, size in sizes:
            block = r[offset:offset + size]; offset += size
            result[('left_' if side == 0 else 'right_') + name] = float(block @ block)
    block = r[offset:offset + ref.n]; offset += ref.n
    result['previous_pose'] = float(block @ block)
    extra = r[offset:]
    assert len(extra) == 2 * int(target['contact'].sum())
    result['geometry_side'] = float(extra[::2] @ extra[::2])
    result['surface_gap'] = float(extra[1::2] @ extra[1::2])
    result['collision_penalty'] = float(.5 * ref.cfg.collision_weight * np.sum(np.maximum(-dist-ref.cfg.collision_tolerance, 0.)**2))
    result['total'] = sum(result.values())
    assert abs(sum(v for k, v in result.items() if k not in ('total', 'collision_penalty')) - float(r @ r)) < 1e-8
    return result


def replay(ref, z, data, body, init, Q, t, names):
    previous = Q[t-1].copy(); q = init[t].copy(); q[ref.qadr] = previous[ref.qadr]
    target = target_at(ref, z, data, t)
    original_solve = core.solve_clarabel_qp_step
    capture = {}
    def record_solve(*args, **kwargs):
        x, info = original_solve(*args, **kwargs)
        capture['x'] = x.copy()
        return x, info
    core.solve_clarabel_qp_step = record_solve
    records = []
    try:
        for it in range(ref.cfg.max_iterations):
            r, J = ref.residuals(q, target, body[t], init[t], previous)
            dist, G, ids = ref.collisions(q)
            before = cost_parts(ref, r, target, dist)
            dq, info = ref.subproblem(q, r, J, dist, G, previous)
            slack = capture['x'][ref.n:]
            predicted = dist + G @ dq
            order = np.argsort(predicted)[:6]
            rows = [dict(**pair_info(ref, ids[i]), before_distance_mm=float(dist[i])*1000,
                         predicted_distance_mm=float(predicted[i])*1000, slack_mm=float(slack[i])*1000,
                         linear_inequality_residual_mm=float(-ref.cfg.collision_tolerance-slack[i]-predicted[i])*1000)
                    for i in order]
            chosen = None; trials = []
            for alpha in (1., .5, .25, .125):
                trial = q.copy(); trial[ref.qadr] += alpha * dq
                rr, _ = ref.residuals(trial, target, body[t], init[t], previous)
                dd, _, ii = ref.collisions(trial, derivatives=False)
                after = cost_parts(ref, rr, target, dd)
                actual = {int(k): float(v) for k, v in zip(ii, dd)}
                trials.append(dict(alpha=alpha, costs=after, collisions=coll_summary(ref, dd, ii),
                                   monitored_actual_distance_mm={str(ids[i]):actual.get(int(ids[i]), ref.cfg.collision_detect)*1000 for i in order}))
                if after['total'] <= before['total'] + 1e-9:
                    chosen = trial
                    break
            records.append(dict(iteration=it, solver=info, before_costs=before, qp_worst_rows=rows,
                                trials=trials, accepted=chosen is not None,
                                accepted_alpha=alpha if chosen is not None else 0., step_norm=float(np.linalg.norm(dq))))
            if chosen is None:
                break
            q = chosen
            if it+1 >= ref.cfg.min_iterations and np.linalg.norm(alpha*dq) < ref.cfg.convergence_step:
                break
    finally:
        core.solve_clarabel_qp_step = original_solve
    delta = q[ref.qadr]-previous[ref.qadr]
    rate_bounds = np.flatnonzero(np.abs(delta) >= ref.cfg.joint_rate_limit-1e-5)
    joint_bounds = np.flatnonzero(np.minimum(q[ref.qadr]-ref.lo, ref.hi-q[ref.qadr]) < 1e-5)
    return dict(frame=t, time_seconds=t/30, replay_max_abs_qpos_difference=float(np.max(np.abs(q-Q[t]))),
                rate_bound_joints=[dict(name=names[i], change_deg=float(np.degrees(delta[i]))) for i in rate_bounds],
                joint_limit_joints=[names[i] for i in joint_bounds], steps=records)


def audit(seq):
    start = time.perf_counter()
    folder = DEST / seq; folder.mkdir(parents=True, exist_ok=True)
    source = destination(seq, 'geometry')
    original_sha = sha(source / 'final_v2.npz')
    info, z, data, ref, body, init = load(seq, 'geometry')
    Q = np.load(source / 'final_v2.npz')['qpos']; nframes = len(Q)
    saved = json.loads((source / 'iterations.json').read_text())
    names = [mujoco.mj_id2name(ref.m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ref.joints]
    torso = mujoco.mj_name2id(ref.m, mujoco.mjtObj.mjOBJ_BODY, 'torso_link')
    assert torso >= 0
    fingers = np.concatenate(ref.finger_columns); arms = np.concatenate(ref.arm_columns)
    a = Q[:, ref.qadr]
    velocity = np.vstack([np.zeros((1, ref.n)), np.diff(a, axis=0)]) * 30
    acceleration = np.vstack([np.zeros((2, ref.n)), np.diff(a, n=2, axis=0)]) * 900
    timeline = []; prev_pose = None
    # FK-only full-clip timeline: inexpensive, no solves and no GPU.
    for t in range(nframes):
        ref.fk(Q[t]); R = rot(Q[t, -4:]); xyz = Q[t, -7:-4]
        p = (ref.d.xpos[torso]-xyz) @ R
        orientation = R.T @ ref.d.xmat[torso].reshape(3, 3)
        if prev_pose is None: relative_mm = relative_deg = 0.
        else:
            relative_mm = float(np.linalg.norm(p-prev_pose[0])*1000)
            relative_deg = angle(prev_pose[1], orientation)
        prev_pose = (p.copy(), orientation.copy())
        timeline.append(dict(frame=t, seconds=t/30,
            torso_object_translation_change_mm=relative_mm,
            torso_object_rotation_change_deg=relative_deg,
            max_finger_change_deg=float(np.degrees(np.max(np.abs(velocity[t, fingers]))/30)),
            max_arm_change_deg=float(np.degrees(np.max(np.abs(velocity[t, arms]))/30)),
            max_finger_acceleration_rad_s2=float(np.max(np.abs(acceleration[t, fingers]))),
            max_arm_acceleration_rad_s2=float(np.max(np.abs(acceleration[t, arms]))),
            max_qp_slack_mm=max(s['slack_max_m'] for s in saved[t]['steps'])*1000,
            iterations=saved[t]['iterations'], linesearch_stopped=saved[t]['linesearch_stopped']))
    events = range(102, 121) if 'organizer' in seq else range(70, 91)
    detailed = []
    for t in events:
        q = init[t].copy(); q[ref.qadr] = Q[t-1, ref.qadr]
        target = target_at(ref, z, data, t)
        previous_target = target_at(ref, z, data, t-1)
        target = target_at(ref, z, data, t)
        d0, _, ids0 = ref.collisions(q, derivatives=False)
        warm_col = coll_summary(ref, d0, ids0)
        ref.fk(q); warm_tips = ref.d.geom_xpos[ref.tipgeom].copy()
        ref.fk(Q[t-1]); previous_tips = ref.d.geom_xpos[ref.tipgeom].copy()
        R0 = rot(Q[t-1, -4:]); R1 = rot(Q[t, -4:])
        previous_local = (previous_tips-Q[t-1, -7:-4]) @ R0
        warm_local = (warm_tips-Q[t, -7:-4]) @ R1
        # Rigidly transport the ENTIRE previous robot with the object. This is
        # only a diagnostic ghost (not a valid body/foot-constrained solution).
        ghost = Q[t-1].copy(); ghost[-7:] = Q[t, -7:]
        ghost[:3] = Q[t, -7:-4] + R1 @ R0.T @ (Q[t-1, :3]-Q[t-1, -7:-4])
        ghost[3:7] = Rotation.from_matrix(R1 @ R0.T @ rot(Q[t-1, 3:7])).as_quat()[[3, 0, 1, 2]]
        dg, _, ig = ref.collisions(ghost, derivatives=False)
        d1, _, ids1 = ref.collisions(Q[t], derivatives=False)
        local0 = (z['hand_kp_scaled'][t-1]-z['object_pose_mujoco'][t-1, :3]) @ data['object_rotation'][t-1]
        local1 = (z['hand_kp_scaled'][t]-z['object_pose_mujoco'][t, :3]) @ data['object_rotation'][t]
        graph_change = [float(np.linalg.norm(target['graphs'][s][0]-previous_target['graphs'][s][0])) for s in (0, 1)]
        entry = dict(**timeline[t], warm_collisions=warm_col, final_collisions=coll_summary(ref, d1, ids1),
            diagnostic_rigid_transport_collisions=coll_summary(ref, dg, ig),
            fixed_body_object_change_displaces_tip_mm=(np.linalg.norm(warm_local-previous_local, axis=-1)*1000).tolist(),
            source_contact_mask_changes=int(np.count_nonzero(data['contact'][t] != data['contact'][t-1])),
            source_hand_max_local_change_mm=float(np.linalg.norm(local1-local0, axis=-1).max()*1000),
            source_anchor_max_change_mm=float(np.linalg.norm(data['anchor_local'][t]-data['anchor_local'][t-1], axis=-1).max()*1000),
            graph_matrix_change_norm=graph_change,
            largest_finger_change=dict(name=names[fingers[np.argmax(np.abs(velocity[t, fingers]))]],
                                      deg=float(np.degrees(velocity[t, fingers[np.argmax(np.abs(velocity[t, fingers]))]])/30)))
        detailed.append(entry)
        print(seq, t, 'warm/final mm', round(warm_col['object']['depth_mm'], 3), round(entry['final_collisions']['object']['depth_mm'], 3), flush=True)
    replay_frames = [110, 111, 112, 113, 114, 115] if 'organizer' in seq else [75, 76, 81, 82, 83]
    replays = []
    for t in replay_frames:
        result = replay(ref, z, data, body, init, Q, t, names)
        replays.append(result)
        print(seq, 'replay', t, 'difference', result['replay_max_abs_qpos_difference'], flush=True)
    write(folder / 'frame_replays.json', replays)
    write(folder / 'event_frames.json', detailed)
    with (folder / 'timeline.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(timeline[0])); writer.writeheader(); writer.writerows(timeline)
    with (folder / 'event_summary.csv').open('w') as f:
        cols = ['frame', 'seconds', 'torso_object_translation_change_mm', 'torso_object_rotation_change_deg',
                'max_finger_change_deg', 'max_arm_change_deg', 'max_qp_slack_mm', 'warm_object_depth_mm', 'final_object_depth_mm']
        writer = csv.DictWriter(f, fieldnames=cols); writer.writeheader()
        for e in detailed:
            row = {k:e[k] for k in cols if k in e}
            row.update(warm_object_depth_mm=e['warm_collisions']['object']['depth_mm'], final_object_depth_mm=e['final_collisions']['object']['depth_mm'])
            writer.writerow(row)
    times = np.arange(nframes)/30
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    axes[0].plot(times, [x['torso_object_translation_change_mm'] for x in timeline], label='Torso vs object displacement / frame')
    axes[0].set_ylabel('mm / frame'); axes[0].legend(loc='upper left')
    et = [x['seconds'] for x in detailed]
    axes[1].plot(et, [x['warm_collisions']['object']['depth_mm'] for x in detailed], '.-', label='Before arm/finger solve')
    axes[1].plot(et, [x['final_collisions']['object']['depth_mm'] for x in detailed], '.-', label='Saved result')
    axes[1].axhline(1, color='gray', ls=':', label='1 mm tolerance before penalized slack')
    axes[1].set_ylabel('Max object penetration (mm)'); axes[1].legend(loc='upper left')
    axes[2].plot(times, np.degrees(velocity[:, fingers])/30, alpha=.45, lw=.8)
    axes[2].axhline(np.degrees(.2), color='black', ls=':'); axes[2].axhline(-np.degrees(.2), color='black', ls=':')
    axes[2].set_ylabel('Finger joint change (deg/frame)')
    axes[3].plot(times, [x['max_qp_slack_mm'] for x in timeline], label='Largest QP collision slack in frame')
    axes[3].set_ylabel('Slack (mm)'); axes[3].set_xlabel('Source time (seconds)'); axes[3].legend(loc='upper left')
    for ax in axes: ax.grid(alpha=.2); ax.set_xlim(2.3, 4.15)
    window = [x for x in timeline if 2.3 <= x['seconds'] <= 4.15]
    axes[0].set_ylim(0, 1.15*max(x['torso_object_translation_change_mm'] for x in window))
    axes[3].set_ylim(0, 1.15*max(x['max_qp_slack_mm'] for x in window))
    fig.suptitle(seq + '\nGeometry side; signed recovery OFF; thumb/index multiplier 1')
    fig.savefig(folder / 'jitter_timeline.png', dpi=160); plt.close(fig)
    assert sha(source / 'final_v2.npz') == original_sha
    result = dict(sequence=seq, trajectory_sha256=original_sha, settings=vars(ref.cfg),
        original_outputs_unchanged=True, frame_indices='zero-based, 30 Hz',
        qp_status_counts=dict(Counter(s['status'] for x in saved for s in x['steps'])),
        replay_max_abs_qpos_difference=max(x['replay_max_abs_qpos_difference'] for x in replays),
        replay_frames=replay_frames, original_iteration_event_statuses=dict(Counter(s['status'] for t in events for s in saved[t]['steps'])),
        elapsed_seconds=time.perf_counter()-start,
        limitations=['Rigid-transport ghost is a diagnostic only: it does not preserve the prescribed body or ground contact.',
                     'Penetration uses the existing registered arm/hand object, ground and self collision pairs.',
                     'No solver settings, objectives, trajectories or evaluation definitions were changed.'])
    write(folder / 'summary.json', result)
    return result


if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    write(OUT / 'selected_variant.json', dict(variant='geometry', signed_recovery=False,
          thumb_index_contact_multiplier=1, hard_side_constraint=False,
          reason='User selected geometry side with recovery off and rejected thumb/index x4 motion quality.'))
    write(DEST / 'manifest.json', dict(variant='geometry', script=str(__file__), script_sha256=sha(__file__),
                                     cpu_threads=1, cuda=False, sequences=list(COHORT)))
    write(DEST / 'summary.json', [audit(seq) for seq in COHORT])
