"""Local collision-only feasibility and derivative audit; no trajectory edits."""
from diagnose_jitter import *
from scipy import sparse
import clarabel


def minimum_slack(ref, q, previous, distance, G, rate=True):
    n = ref.n
    lower, upper = ref.lo-q[ref.qadr], ref.hi-q[ref.qadr]
    if rate:
        lower = np.maximum(lower, previous[ref.qadr]-ref.cfg.joint_rate_limit-q[ref.qadr])
        upper = np.minimum(upper, previous[ref.qadr]+ref.cfg.joint_rate_limit-q[ref.qadr])
    I = sparse.eye(n, format='csc'); Z = sparse.csc_matrix((n, 1)); k = len(distance)
    A = sparse.vstack([sparse.hstack([I, Z]), sparse.hstack([-I, Z]),
        sparse.hstack([-sparse.csc_matrix(G), -np.ones((k, 1))]),
        sparse.csc_matrix(([ -1. ], ([0], [n])), shape=(1, n+1))], format='csc')
    b = np.r_[upper, -lower, distance+ref.cfg.collision_tolerance, 0.]
    cones = [clarabel.NonnegativeConeT(len(b))]
    A = sparse.vstack([A, sparse.csc_matrix((1, n+1)), sparse.hstack([-I, Z])], format='csc')
    b = np.r_[b, ref.cfg.trust_radius, np.zeros(n)]
    cones.append(clarabel.SecondOrderConeT(n+1))
    P = sparse.diags(np.r_[np.full(n, 1e-8), 0.], format='csc')
    x, info = core.solve_clarabel_qp_step(P, np.r_[np.zeros(n), 1.], A, b, cones, ref.solver_settings)
    return dict(minimum_common_linearized_slack_mm=max(0., float(x[-1]))*1000,
                step_norm=float(np.linalg.norm(x[:n])), solver=info)


def run(seq):
    info, z, data, ref, body, init = load(seq, 'geometry')
    Q = np.load(destination(seq, 'geometry')/'final_v2.npz')['qpos']
    frames = [110, 111, 112, 113, 114] if 'organizer' in seq else [75, 81, 82]
    rows = []
    for t in frames:
        for mode in ('warm', 'final'):
            previous = Q[t-1]
            q = Q[t].copy() if mode == 'final' else init[t].copy()
            if mode == 'warm': q[ref.qadr] = previous[ref.qadr]
            dist, G, ids = ref.collisions(q)
            row = dict(frame=t, mode=mode, local_feasibility={})
            for categories in ('all', 'object'):
                mask = np.ones(len(ids), dtype=bool) if categories == 'all' else ref.pairs[ids, 2] == 0
                for rate in (True, False):
                    row['local_feasibility'][categories+('_rate_bound' if rate else '_no_rate_bound')] = minimum_slack(ref, q, previous, dist[mask], G[mask], rate)
            # Finite-difference only the three deepest object pairs, all active DoFs.
            idx = np.flatnonzero(ref.pairs[ids, 2] == 0)
            idx = idx[np.argsort(dist[idx])[:3]]
            checks = []
            for i in idx:
                ga, gb, _ = map(int, ref.pairs[ids[i]])
                numeric = np.zeros(ref.n)
                for col, adr in enumerate(ref.qadr):
                    plus = q.copy(); minus = q.copy(); plus[adr] += 1e-6; minus[adr] -= 1e-6
                    ref.fk(plus); dp = ref.queries.query(ref.d, ga, gb)[0]
                    ref.fk(minus); dm = ref.queries.query(ref.d, ga, gb)[0]
                    numeric[col] = (dp-dm)/2e-6
                checks.append(dict(**pair_info(ref, ids[i]), signed_distance_mm=float(dist[i])*1000,
                                   max_gradient_error_m_per_rad=float(np.max(np.abs(numeric-G[i]))),
                                   analytic_gradient_norm=float(np.linalg.norm(G[i])),
                                   finite_difference_gradient_norm=float(np.linalg.norm(numeric))))
            row['derivative_checks'] = checks
            rows.append(row)
            print(seq, t, mode, {k:round(v['minimum_common_linearized_slack_mm'], 4) for k,v in row['local_feasibility'].items()}, flush=True)
    write(DEST/seq/'constraint_audit.json', dict(rows=rows,
        interpretation='This is the minimum common slack of the LOCAL LINEARIZED collision model with the original 0.45 rad trust region. It is not a certificate of global nonlinear reachability. Hand objectives are omitted only in this diagnostic QP. Original trajectories and settings are untouched.'))


def direction_history(seq):
    info, z, data, ref, body, init = load(seq, 'geometry')
    Q = np.load(destination(seq, 'geometry')/'final_v2.npz')['qpos']
    records, last = [], {}
    for t in range(108, 115):
        target_at(ref, z, data, t)
        q = init[t].copy(); q[ref.qadr] = Q[t-1, ref.qadr]
        ref.fk(q); R = rot(q[-4:]); rows = []
        for pair, finger in [(282, 1), (303, 1), (324, 1), (364, 2), (385, 2), (387, 2), (408, 2)]:
            ga, gb, _ = map(int, ref.pairs[pair])
            distance, pa, pb, normal = ref.queries.query(ref.d, ga, gb)
            local = R.T @ normal; intended = data['direction_local'][t, 0, finger]
            rows.append(dict(**pair_info(ref, pair), distance_mm=float(distance)*1000,
                escape_direction_object_frame=local.tolist(), source_side_direction_object_frame=intended.tolist(),
                escape_dot_source_direction=float(local@intended), direction_change_deg=None if pair not in last else
                float(np.degrees(np.arccos(np.clip(last[pair]@local, -1, 1))))))
            last[pair] = local.copy()
        records.append(dict(frame=t, seconds=t/30, initial_collision_queries=rows))
    frozen = np.setdiff1d(np.arange(ref.m.nq), ref.qadr)
    write(DEST/seq/'collision_direction_history.json', dict(frames=records,
          frozen_body_equals_omni=bool(np.array_equal(Q[:, frozen], body[:, frozen])),
          note='Exact original warm starts before SQP; escape directions are signed convex-distance gradients.'))


if __name__ == '__main__':
    for seq in COHORT: run(seq)
    direction_history(COHORT[0])
