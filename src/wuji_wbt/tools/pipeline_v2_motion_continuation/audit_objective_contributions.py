"""Measure weighted residual costs on saved trajectories; never rerun retargeting.

Percentages describe the final-pose nonlinear merit, summed over all frames.
They are not causal influence or QP multiplier percentages. The collision entry
uses measured depth beyond tolerance, matching the nonlinear line-search merit,
not the saved linearized QP's slack variables. Damping is a local-step cost and
is reported separately from recorded QP proposals.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1",
                  OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

import csv
import time
import numpy as np
from config import OUT, CODE, COHORT, destination, read, write, sha
from solver import load, target_at

DEST = OUT / "objective_contributions"
TERMS = (
    "interaction_mesh", "human_fingertip_tracking", "legacy_point_attraction",
    "finger_bone_directions", "palm_alignment", "omni_arm_prior",
    "finger_initialization_prior", "temporal_smoothness", "geometry_side",
    "regional_surface_gap", "collision_object", "collision_ground",
    "collision_self",
)


def decompose(ref, residual, jacobian, target, previous, distance, gradient, ids):
    costs = dict.fromkeys(TERMS, 0.)
    gradients = {name: np.zeros(ref.n) for name in TERMS}
    offset = 0

    def consume(name, size):
        nonlocal offset
        r = residual[offset:offset+size]
        J = jacobian[offset:offset+size]
        offset += size
        costs[name] += float(r @ r)
        gradients[name] += 2 * J.T @ r

    for side in range(2):
        for name, size in (
            ("interaction_mesh", len(target["graphs"][side][0])*3),
            ("human_fingertip_tracking", 15),
            ("legacy_point_attraction", int(target["contact"][side].sum())*3),
            ("finger_bone_directions", 45), ("palm_alignment", 6),
            ("omni_arm_prior", 7), ("finger_initialization_prior", 20),
        ):
            consume(name, size)
    if previous is not None:
        consume("temporal_smoothness", ref.n)
    active = int(target["contact"].sum())
    assert len(residual)-offset == 2*active
    for name, parity in (("geometry_side", 0), ("regional_surface_gap", 1)):
        r = residual[offset+parity::2]
        J = jacobian[offset+parity::2]
        costs[name] = float(r @ r)
        gradients[name] = 2 * J.T @ r
    np.testing.assert_allclose(sum(costs.values()), residual @ residual,
                               atol=1e-10, rtol=1e-10)
    excess = np.maximum(-distance-ref.cfg.collision_tolerance, 0.)
    for category, name in enumerate(("collision_object", "collision_ground",
                                      "collision_self")):
        mask = ref.pairs[ids, 2] == category
        costs[name] = .5*ref.cfg.collision_weight*float(excess[mask] @ excess[mask])
        gradients[name] = -ref.cfg.collision_weight * gradient[mask].T @ excess[mask]
    expected = float(residual @ residual +
                     .5*ref.cfg.collision_weight*(excess @ excess))
    np.testing.assert_allclose(sum(costs.values()), expected, atol=1e-10, rtol=1e-10)
    return costs, {name: float(np.linalg.norm(g)) for name, g in gradients.items()}


def audit(seq, variant):
    started = time.perf_counter()
    info, z, data, ref, body, init = load(seq, variant)
    path = destination(seq, variant)/"final_v2.npz"
    before = sha(path)
    saved_run = read(path.parent/"run.json")
    assert vars(ref.cfg) == saved_run["settings"]
    with np.load(path) as archive:
        Q = archive["qpos"]
    rows, total_costs, total_gradients = [], dict.fromkeys(TERMS, 0.), dict.fromkeys(TERMS, 0.)
    for t, q in enumerate(Q):
        previous = Q[t-1] if t else None
        target = target_at(ref, z, data, t)
        residual, J = ref.residuals(q, target, body[t], init[t], previous)
        distance, G, ids = ref.collisions(q, derivatives=True)
        costs, norms = decompose(ref, residual, J, target, previous, distance, G, ids)
        total = sum(costs.values())
        rows.append(dict(sequence=seq, variant=variant, frame=t, seconds=t/30,
                         source_contact_fingers=int(target["contact"].sum()),
                         total_weighted_cost=total, **costs))
        for name in TERMS:
            total_costs[name] += costs[name]
            total_gradients[name] += norms[name]
        if t % 50 == 0:
            print(seq, variant, t+1, "/", len(Q), flush=True)
    total = sum(total_costs.values())
    normsum = sum(total_gradients.values())
    steps = [step for frame in read(path.parent/"iterations.json")
             for step in frame["steps"]]
    damping = np.array([ref.cfg.damping*s["step_norm"]**2 for s in steps])
    result = dict(
        sequence=seq, variant=variant, frames=len(Q),
        trajectory_sha256=before, total_weighted_cost=total,
        mean_weighted_cost=total/len(Q),
        terms={name: dict(sum_weighted_cost=total_costs[name],
                         mean_weighted_cost=total_costs[name]/len(Q),
                         percent_of_final_pose_merit=100*total_costs[name]/total,
                         mean_gradient_norm=total_gradients[name]/len(Q),
                         percent_of_sum_of_term_gradient_norms=
                         100*total_gradients[name]/normsum if normsum else 0.)
               for name in TERMS},
        numerical_damping=dict(
            weight=ref.cfg.damping, proposed_steps=len(steps),
            mean_cost_per_proposed_qp_step=float(damping.mean()),
            max_cost_per_proposed_qp_step=float(damping.max()),
            included_in_final_pose_merit=False,
            note="0.001 times norm(dq)^2 for raw QP proposals, before line search."),
        elapsed_seconds=time.perf_counter()-started,
    )
    assert sha(path) == before
    dest = DEST / variant
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/f"{seq}_per_frame.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(dest/f"{seq}.json", result)
    return result


def main():
    started = time.perf_counter()
    protocol = read(OUT/"protocol.json")
    # Guard against accidentally evaluating a different implementation/input.
    for group in ("source_sha256", "input_sha256", "target_sha256"):
        for path, expected in protocol[group].items():
            assert sha(path) == expected, (group, path)
    records = [audit(seq, variant) for seq in COHORT
               for variant in ("baseline", "motion_init")]
    table = []
    for record in records:
        for term, measurements in record["terms"].items():
            table.append(dict(sequence=record["sequence"], variant=record["variant"],
                              term=term, **measurements))
    with (DEST/"term_contributions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write(DEST/"summary.json", dict(
        records=records, audit_script_sha256=sha(CODE/"audit_objective_contributions.py"),
        definition="100 * sum_frames(weighted_term_cost) / sum_frames(all_weighted_term_costs)",
        evaluated_at="Saved final pose of every frame; same source targets, previous output pose, and model.",
        collision="Measured excess-depth penalty from nonlinear merit; not stored QP slack.",
        gradient="Unconstrained per-term gradient norms at final poses; can cancel and do not attribute causal influence.",
        damping="Local step regularizer, excluded from final-pose percentages and reported separately.",
        constraints="Hard joint/rate/trust limits have no additive cost percentage.",
        invariants="Source hashes verified; no retargeting, trajectories, objective definitions, or metrics changed.",
        resources="Single CPU thread, CUDA disabled.",
        elapsed_seconds=time.perf_counter()-started,
    ))
    print("Objective audit completed:", DEST, flush=True)


if __name__ == "__main__":
    main()
