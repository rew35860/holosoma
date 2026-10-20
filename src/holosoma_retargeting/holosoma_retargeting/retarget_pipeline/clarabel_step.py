"""Unscaled Clarabel steps with explicit precision and checked stall recovery.

Feasibility alone is insufficient for a stalled iterate. For supported cones,
project its dual variables into the dual cone and compute a QP dual lower bound
using the positive-definite damped Hessian. Accept stalled iterates only when
original-unit feasibility and this independently computed gap are both small.
No coordinate clipping or objective scaling is performed.
"""
from __future__ import annotations

import clarabel
import numpy as np
from scipy import sparse, linalg

POLICY = 'explicit-tolerances-and-certified-fallback-v1'
ACCEPTED_STATUSES = {'Solved', 'AlmostSolved'}
RECOVERABLE_STATUSES = {'InsufficientProgress', 'MaxIterations'}
FEASIBILITY_TOLERANCE = 1e-6


def make_settings(max_iter=100):
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = max_iter
    settings.tol_gap_abs = 1e-6
    settings.tol_gap_rel = 1e-6
    settings.tol_feas = 1e-7
    # Keep every reduced_tol_* value at the installed Clarabel default.
    return settings


def cone_violation(vector, cones):
    """Maximum violation of actual affine cone coordinates (not solver pres)."""
    offset, violation = 0, 0.
    for cone in cones:
        dim = int(cone.dim)
        part = vector[offset:offset+dim]
        if len(part) != dim:
            raise ValueError('Cone dimensions do not match the vector')
        if isinstance(cone, clarabel.NonnegativeConeT):
            violation = max(violation, float(np.max(-part, initial=0.)))
        elif isinstance(cone, clarabel.SecondOrderConeT):
            violation = max(violation, float(np.linalg.norm(part[1:])-part[0]))
        elif isinstance(cone, clarabel.ZeroConeT):
            violation = max(violation, float(np.max(abs(part), initial=0.)))
        else:
            raise ValueError(f'Unsupported checked cone: {type(cone).__name__}')
        offset += dim
    if offset != len(vector):
        raise ValueError('Cone dimensions do not exhaust the vector')
    return violation


def project_dual(vector, cones):
    result = vector.copy()
    offset = 0
    for cone in cones:
        dim = int(cone.dim)
        part = result[offset:offset+dim]
        if isinstance(cone, clarabel.NonnegativeConeT):
            part[:] = np.maximum(part, 0.)
        elif isinstance(cone, clarabel.SecondOrderConeT):
            t, radius = float(part[0]), float(np.linalg.norm(part[1:]))
            if radius <= t:
                pass
            elif radius <= -t:
                part[:] = 0.
            else:
                new_t = .5*(radius+t)
                part[1:] *= new_t/radius
                part[0] = new_t
        elif not isinstance(cone, clarabel.ZeroConeT):
            raise ValueError(f'Unsupported checked cone: {type(cone).__name__}')
        # The dual of the zero cone is unconstrained.
        offset += dim
    return result


def assess_iterate(P, linear, A, b, cones, result, settings):
    status = str(result.status)
    audit = dict(status=status, accepted=False, accepted_feasible_fallback=False,
                 numerical_objective_scale=1., solver_policy=POLICY,
                 feasibility_tolerance=FEASIBILITY_TOLERANCE)
    try:
        x = np.asarray(result.x, dtype=float)
    except (AttributeError, ValueError, TypeError):
        audit['rejection_reason'] = 'Missing or invalid primal iterate'
        return audit
    if x.shape != linear.shape or not np.isfinite(x).all():
        audit['rejection_reason'] = 'Nonfinite or wrong-sized primal iterate'
        return audit
    slack = b-A@x
    if not np.isfinite(slack).all():
        audit['rejection_reason'] = 'Nonfinite original-unit constraints'
        return audit
    violation = cone_violation(slack, cones)
    audit['maximum_linear_constraint_violation'] = violation
    audit['maximum_cone_constraint_violation'] = violation
    audit['objective_original'] = float(.5*x@(P@x)+linear@x)
    if not np.isfinite(audit['objective_original']):
        audit['rejection_reason'] = 'Nonfinite original objective'
        return audit
    if violation > FEASIBILITY_TOLERANCE:
        audit['rejection_reason'] = 'Original-unit cone infeasibility'
        return audit
    if status in ACCEPTED_STATUSES:
        audit['accepted'] = True
        return audit
    if status not in RECOVERABLE_STATUSES:
        audit['rejection_reason'] = 'Status is not eligible for numerical-stall recovery'
        return audit
    try:
        z = np.asarray(result.z, dtype=float)
    except (AttributeError, ValueError, TypeError):
        z = np.array([])
    if z.shape != b.shape or not np.isfinite(z).all():
        audit['rejection_reason'] = 'Missing, nonfinite or wrong-sized dual certificate'
        return audit
    z = project_dual(z, cones)
    u = linear+A.T@z
    try:
        # P includes damping and positive slack penalties; do not assume PD if
        # a future caller changes that construction. Cholesky must succeed.
        factor = linalg.cho_factor(P.toarray(), lower=True, check_finite=True)
        inverse_u = linalg.cho_solve(factor, u, check_finite=True)
    except (ValueError, linalg.LinAlgError):
        audit['rejection_reason'] = 'Cannot certify a positive-definite QP Hessian'
        return audit
    dual_bound = float(-.5*u@inverse_u-b@z)
    primal = audit['objective_original']
    gap = primal-dual_bound
    scale = max(1., abs(primal), abs(dual_bound))
    limit = max(float(settings.tol_gap_abs), float(settings.tol_gap_rel)*scale)
    audit.update(independent_dual_lower_bound=dual_bound, independent_gap=gap,
                 independent_gap_limit=limit,
                 projected_dual_stationarity_inf=float(np.linalg.norm(P@x+u, np.inf)))
    # A substantially negative gap signals an unreliable near-feasible bound.
    # Permit only floating-point-level cancellation, not a feasibility shortcut.
    if not np.isfinite([primal, dual_bound, gap]).all() or gap < -1e-9*scale or gap > limit:
        audit['rejection_reason'] = 'Independent optimality gap is not sufficiently small'
        return audit
    audit.update(accepted=True, accepted_feasible_fallback=True)
    return audit


def solve_clarabel_qp_step(P, linear, A, b, cones, settings=None):
    """Solve once, inspect the unmodified iterate, and return it without clipping."""
    settings = make_settings() if settings is None else settings
    solver = clarabel.DefaultSolver(sparse.triu(P).tocsc(), linear, A, b, cones, settings)
    result = solver.solve()
    audit = assess_iterate(P, linear, A, b, cones, result, settings)
    details = solver.get_info()
    audit.update(iterations=int(result.iterations), solver_seconds=float(result.solve_time),
                 reported_primal_residual=float(details.res_primal),
                 reported_dual_residual=float(details.res_dual),
                 reported_gap_abs=float(details.gap_abs), reported_gap_rel=float(details.gap_rel))
    if not audit['accepted']:
        raise RuntimeError(f'Clarabel step rejected: {audit}')
    return np.asarray(result.x, dtype=float), audit
