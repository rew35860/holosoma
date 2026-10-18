"""Numerical regression checks for the v2 optimization, including collision signs."""
from pathlib import Path
import json
import sys
import numpy as np
import mujoco
import cvxpy as cp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_v2 import CoupledRefiner, Settings

ROOT = Path('/home/natcha/Downloads')
BATCH = ROOT/'holosoma/eval_out/ref_candidates/batch_20261011_current_120'
OUT = ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_pilot_20261018'


def check_qp():
    r = CoupledRefiner.__new__(CoupledRefiner)
    r.n = 3;r.qadr = np.arange(3);r.lo = np.full(3, -1.);r.hi = np.full(3, 1.)
    r.cfg = Settings()
    import clarabel
    r.solver_settings = clarabel.DefaultSettings();r.solver_settings.verbose = False
    J = np.array([[1., 2., 0.], [0., 1., .5], [.4, 0., 1.]])
    e = np.array([-.7, .2, .1]);G = np.array([[1., -.2, 0.], [0., 0., 1.]])
    dist = np.array([-.01, .02]);q = np.zeros(3)
    actual, _ = r.subproblem(q, e, J, dist, G, None)
    dq = cp.Variable(3);s = cp.Variable(2, nonneg=True)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(e+J@dq)+r.cfg.damping*cp.sum_squares(dq)
                                    +r.cfg.collision_weight/2*cp.sum_squares(s)),
                         [dq >= -1, dq <= 1, cp.norm(dq) <= r.cfg.trust_radius,
                          dist+G@dq+s >= -r.cfg.collision_tolerance])
    problem.solve(solver='CLARABEL')
    error = float(np.max(np.abs(actual-dq.value)))
    assert error < 2e-5, error
    return error


def check_geometry():
    seq = 'carrying_dining_chair_with_both_hands_and_back-660'
    job = json.loads((BATCH/'jobs'/f'{seq}.json').read_text())
    m = mujoco.MjModel.from_xml_path(job['model_xml'])
    r = CoupledRefiner(m, 'dining_chair')
    q = np.load(job['final_npz'])['qpos'][150].copy()
    r.fk(q);p, J = r.keypoints()
    forward = mujoco.MjData(m);forward.qpos[:] = q;mujoco.mj_forward(m, forward)
    assert np.allclose(r.d.geom_xpos, forward.geom_xpos, atol=1e-12)
    assert np.allclose(r.d.xmat, forward.xmat, atol=1e-12)
    rng = np.random.default_rng(27);v = rng.normal(size=54);v /= np.linalg.norm(v)
    eps = 1e-6
    plus = q.copy();plus[r.qadr] += eps*v;r.fk(plus);pp = r.keypoints(False)
    minus = q.copy();minus[r.qadr] -= eps*v;r.fk(minus);pm = r.keypoints(False)
    error = float(np.max(np.abs((pp-pm)/(2*eps)-np.einsum('hkjn,n->hkj', J, v))))
    assert error < 1e-6, error
    dist, gradient, ids = r.collisions(q)
    # Check native signed-distance derivatives only where the closest feature is smooth.
    errors = []
    for k in np.argsort(np.linalg.norm(gradient, axis=1))[-12:]:
        ga, gb, _ = r.pairs[ids[k]]
        r.fk(plus);a = mujoco.mj_geomDistance(m, r.d, int(ga), int(gb), .1, None)
        r.fk(minus);b = mujoco.mj_geomDistance(m, r.d, int(ga), int(gb), .1, None)
        numerical = (a-b)/(2*eps);analytical = gradient[k]@v
        errors.append(float(abs(numerical-analytical)))
    assert np.median(errors) < 2e-3, errors
    return dict(keypoint_directional_jacobian_max_error=error,
                collision_directional_jacobian_median_error=float(np.median(errors)),
                collision_directional_errors=errors, kinematic_only_matches_forward=True,
                active_dofs=r.n, collision_pair_counts=r.pair_categories)


if __name__ == '__main__':
    results = dict(qp_max_difference_from_cvxpy=check_qp(), geometry=check_geometry(), passed=True)
    (OUT/'numerical_checks.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))
