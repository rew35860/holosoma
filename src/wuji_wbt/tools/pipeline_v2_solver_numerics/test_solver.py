"""Numerical-stall replay and rejection tests for unsafe returned iterates."""
from types import SimpleNamespace
import clarabel
import numpy as np
from scipy import sparse
from config import *

step = module('checked_clarabel_step', RET / 'retarget_pipeline/clarabel_step.py')


def main():
    settings = step.make_settings()
    defaults = clarabel.DefaultSettings()
    reduced = {name: getattr(settings, name) for name in dir(settings) if name.startswith('reduced_tol_')}
    assert all(value == getattr(defaults, name) for name, value in reduced.items())
    tests = []
    P = sparse.eye(1, format='csc')
    q = np.array([-2.])
    A = sparse.csc_matrix([[-1.], [1.]])
    b = np.array([0., 3.])
    cones = [clarabel.NonnegativeConeT(2)]

    def check(name, result, expected, *problem):
        audit = step.assess_iterate(*(problem or (P, q, A, b, cones)), result, settings)
        assert audit['accepted'] == expected, (name, audit)
        tests.append(dict(test=name, passed=True, audit=audit))
        return audit

    def result(x, status='InsufficientProgress', z=(0., 0.)):
        return SimpleNamespace(x=x, z=z, status=status)

    for status in ('InsufficientProgress', 'MaxIterations'):
        audit = check(f'{status}: optimal feasible iterate', result([2.], status), True)
        assert abs(audit['independent_dual_lower_bound'] + 2.) < 1e-12
        check(f'{status}: feasible but suboptimal iterate', result([0.], status), False)
    check('Reject infeasible Solved iterate', result([4.], 'Solved'), False)
    check('Reject infeasible AlmostSolved iterate', result([-0.1], 'AlmostSolved'), False)
    check('Reject infeasible-problem status', result([2.], 'PrimalInfeasible'), False)
    check('Reject nonfinite primal iterate', result([float('nan')]), False)
    check('Reject nonfinite dual certificate', result([2.], z=(float('nan'), 0.)), False)
    check('Reject missing dual certificate', result([2.], z=()), False)
    check('Reject wrong-sized primal iterate', result([1., 2.]), False)
    check('Reject poor dual certificate', result([2.], z=(100., 0.)), False)
    check('Project dual into feasible nonnegative cone', result([2.], z=(-1., -1.)), True)
    check('Reject singular Hessian in fallback', result([2.]), False,
          sparse.csc_matrix((1, 1)), q, A, b, cones)
    # Verify the actual affine SOC coordinates, rather than assuming x itself
    # is the cone vector. Here x=0 has norm zero but b-Ax=(1,2) is infeasible.
    soc_A = sparse.csc_matrix([[0.], [-1.]])
    soc_b = np.array([1., 2.])
    check('Reject affine SOC violation despite zero x', result([0.], 'Solved'), False,
          P, q, soc_A, soc_b, [clarabel.SecondOrderConeT(2)])
    # A real constrained optimum: min .5*x^2-2*x, x<=1. Dual z=1,
    # objective and independently derived dual lower bound are both -1.5.
    audit = check('Active bound with nonzero dual certificate', result([1.], z=(1.,)), True,
                  P, q, sparse.csc_matrix([[1.]]), np.array([1.]), [clarabel.NonnegativeConeT(1)])
    assert abs(audit['independent_dual_lower_bound'] + 1.5) < 1e-12
    # Equality constraint x=1: the zero cone has a free dual variable.
    check('Equality cone dual certificate', result([1.], z=(1.,)), True,
          P, q, sparse.csc_matrix([[1.]]), np.array([1.]), [clarabel.ZeroConeT(1)])

    archive = np.load(ARCHIVED_QP)
    P0, A0 = sparse.csc_matrix(archive['P']), sparse.csc_matrix(archive['A'])
    q0, b0 = archive['linear'], archive['b']
    cones0 = [clarabel.NonnegativeConeT(int(archive['nonnegative_rows'])),
              clarabel.SecondOrderConeT(int(archive['active_dofs'])+1)]
    old = clarabel.DefaultSettings()
    old.verbose = False; old.max_iter = 100; old.tol_gap_abs = 1e-7; old.tol_feas = 1e-7
    old_solver = clarabel.DefaultSolver(sparse.triu(P0).tocsc(), q0, A0, b0, cones0, old)
    stalled = old_solver.solve()
    assert str(stalled.status) == 'InsufficientProgress'
    recovered = step.assess_iterate(P0, q0, A0, b0, cones0, stalled, settings)
    assert recovered['accepted_feasible_fallback'], recovered
    x, current = step.solve_clarabel_qp_step(P0, q0, A0, b0, cones0, settings)
    assert current['status'] == 'Solved' and current['numerical_objective_scale'] == 1.
    assert current['maximum_cone_constraint_violation'] <= 1e-6
    # Direct reference solve must return exactly the step exposed by our helper:
    # there is no clipping or other silent coordinate repair.
    direct_solver = clarabel.DefaultSolver(sparse.triu(P0).tocsc(), q0, A0, b0, cones0, settings)
    direct = direct_solver.solve()
    assert np.array_equal(x, np.asarray(direct.x))
    write(OUT / 'solver_tests.json', dict(passed=True, synthetic_tests=tests,
        clarabel_version=clarabel.__version__, reduced_tolerances=reduced,
        archived_qp=str(ARCHIVED_QP), archived_qp_sha256=sha(ARCHIVED_QP),
        old_status=str(stalled.status), old_iterations=stalled.iterations,
        old_reported_primal_residual=old_solver.get_info().res_primal,
        old_iterate_certified=recovered, new_result=current,
        max_step_difference_from_direct_solve=float(np.max(abs(x-direct.x))),
        tolerance_note='Gap tolerances concern objective accuracy, not joint-angle accuracy.'))
    print(f'{len(tests)} rejection/certificate checks passed; archived failure: {current}', flush=True)


if __name__ == '__main__':
    main()
