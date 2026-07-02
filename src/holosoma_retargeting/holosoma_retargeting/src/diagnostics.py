"""Solver-failure diagnostics for the interaction-mesh retargeter.

Kept out of interaction_mesh_retargeter.py so the retargeting logic stays clean:
this only runs when a CVXPY solve fails, to report *why* (which constraint group
was violated, which geom pairs penetrate, and the escape direction).

`rt` is the retargeter instance; the function reads its geom names, tolerances,
and cached penetration normals just as the original method read `self`.
"""
from __future__ import annotations

import sys

import cvxpy as cp  # type: ignore[import-not-found]


def diagnose_infeasibility(rt, frame_idx, status, groups, phis, phis_sc):
    """Elastic / feasibility relaxation to find which hard-constraint groups
    made the QP infeasible, and by how much.

    Adds one nonnegative slack per inequality group (loosening every
    constraint in that group), keeps the laplacian-definition equality and
    the step-size SOC hard, then minimizes total slack. A group's slack > 0
    means it had to be loosened by that amount to reach feasibility -- i.e.
    that group is (part of) what's violated. Prints a report to stderr and
    returns a one-line summary for the raised exception.
    """
    keep_hard = {"laplacian_def", "step_size"}
    relaxed, slacks = [], {}
    for name, cons in groups.items():
        if not cons:
            continue
        if name in keep_hard:
            relaxed.extend(cons)
            continue
        s = cp.Variable(nonneg=True, name=f"slack_{name}")
        slacks[name] = s
        # cvxpy normalizes any inequality to `c.expr <= 0`; loosen to `<= s`.
        relaxed.extend([c.expr <= s for c in cons])

    violated = {}
    if slacks:
        prob = cp.Problem(cp.Minimize(cp.sum(list(slacks.values()))), relaxed)
        try:
            prob.solve(solver=cp.CLARABEL)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                for name, s in slacks.items():
                    v = float(s.value) if s.value is not None else 0.0
                    if v > 1e-9:
                        violated[name] = v
        except Exception as exc:  # noqa: BLE001
            print(f"[retarget][frame {frame_idx}] elastic relaxation failed: {exc}")

    # Resolve geom ids -> human-readable names so you can see *what* touches
    # what (e.g. "left_hand vs organizer" vs "right_foot vs ground").
    def gname(gid):
        try:
            return rt._geom_names[gid]
        except Exception:  # noqa: BLE001
            return f"geom{gid}"

    # Sorted worst-first (most negative signed distance = deepest overlap).
    pen_sorted = sorted(phis.items(), key=lambda kv: kv[1])
    sc_sorted = sorted(phis_sc.items(), key=lambda kv: kv[1])

    out = ["=" * 72,
           f"[retarget] INFEASIBLE QP at frame {frame_idx} (status={status})",
           f"  constraint groups present: "
           f"{ {k: len(v) for k, v in groups.items() if v} }"]
    if violated:
        out.append("  VIOLATED groups (slack = how much they must be loosened):")
        for name, v in sorted(violated.items(), key=lambda kv: -kv[1]):
            out.append(f"    - {name:16s} needs +{v:.5f}")
    else:
        out.append("  (elastic relaxation found no single group to blame; "
                   "likely a joint conflict between groups)")

    def escape_dir(g1, g2):
        """Escape direction of the ROBOT link, i.e. the direction the
        constraint demands it move to separate. Stored nhat is the motion
        of geom1 relative to geom2; flip if geom1 is the environment."""
        n = getattr(rt, "_last_pen_normals", {}).get((g1, g2))
        if n is None:
            return ""
        if "ground" in gname(g1) or rt.object_name in gname(g1):
            n = -n
        return f"  escape dir = [{n[0]:+.2f} {n[1]:+.2f} {n[2]:+.2f}]"

    out.append(f"  non_penetration pairs (tol = {rt.penetration_tolerance}, "
               f"negative = penetrating; escape dir = direction the robot "
               f"link is constrained to move, world xyz):")
    # Show the worst pairs, plus every pair involving the object: the object
    # pairs are the ones that can block a base lift even when the worst
    # offenders are all feet-vs-ground.
    shown = set()
    for (g1, g2), d in pen_sorted[:5]:
        out.append(f"    {d:+.5f}  {gname(g1)} <-> {gname(g2)}{escape_dir(g1, g2)}")
        shown.add((g1, g2))
    obj_pairs = [
        ((g1, g2), d) for (g1, g2), d in pen_sorted
        if (g1, g2) not in shown
        and (rt.object_name in gname(g1) or rt.object_name in gname(g2))
    ]
    if obj_pairs:
        out.append("    --- robot <-> object pairs ---")
        for (g1, g2), d in obj_pairs:
            out.append(f"    {d:+.5f}  {gname(g1)} <-> {gname(g2)}{escape_dir(g1, g2)}")
    if not pen_sorted:
        out.append("    (none within detection threshold)")

    out.append(f"  self_collision pairs (tol = {rt._self_collision_tolerance}):")
    for key, d in sc_sorted[:5]:
        g1, g2 = key[1], key[2]
        out.append(f"    {d:+.5f}  {gname(g1)} <-> {gname(g2)}")
    if not sc_sorted:
        out.append("    (none within detection threshold)")

    out.append(f"  foot_sticking tol = {rt.foot_sticking_tolerance}, "
               f"step_size = {rt.step_size}")
    out.append("=" * 72)

    # stderr + flush so it lands next to the traceback, not lost in stdout.
    print("\n".join(out), file=sys.stderr, flush=True)

    if violated:
        return "Violated: " + ", ".join(
            f"{k}(+{v:.4f})" for k, v in sorted(violated.items(), key=lambda kv: -kv[1])
        )
    return "No single group isolated; inter-group conflict."
