"""Measure contact terms at one saved frame; no optimization or GPU use.

Translated examples are geometry probes, not retargeted or reachable robot poses.
The saved optimizer targets and evaluation definitions are never modified.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1',
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np
import coal
from config import OUT, COHORT, CODE, write, sha
from solver import load, target_at, mujoco
from geometry import unsigned_query


def main():
    seq = COHORT[0]
    info, z, data, ref, body, init = load(seq, 'hard')
    t, side, finger = 120, 1, 0
    target = target_at(ref, z, data, t)
    pose_file = OUT / 'variants/hard/clips' / seq / 'final_v2.npz'
    q = np.load(pose_file)['qpos'][t]
    ref.fk(q)
    geom = int(ref.tipgeom[side, finger])
    normal_local = data['direction_local'][t, side, finger]
    anchor_local = data['anchor_local'][t, side, finger]
    normal = target['R'] @ normal_local
    anchor = target['R'] @ anchor_local + target['object_pose'][:3]
    rotation = ref.d.geom_xmat[geom].reshape(3, 3).copy()
    position = ref.d.geom_xpos[geom].copy()
    vertices = ref.tip_vertices[geom]
    source_gap = float(data['source_skin_gap'][t, side, finger])
    tau = float(ref.cfg.collision_tolerance)
    object_transform = coal.Transform3s(target['R'], target['object_pose'][:3])
    start, end = data['patch_offsets'][t, side, finger]
    triangles = data['patch_triangles'][start:end]

    def measure(offset_mm):
        p = position + normal * offset_mm / 1000.
        transform = coal.Transform3s(rotation, p)
        gap, _, _ = unsigned_query(ref.queries.shapes[geom], transform,
            target['patches'][side, finger], object_transform)
        distances = []
        for obj in ref.material[geom]:
            result = coal.DistanceResult()
            distance = float(coal.distance(ref.queries.shapes[geom], transform,
                ref.queries.shapes[int(obj)],
                coal.Transform3s(ref.d.geom_xmat[obj].reshape(3, 3), ref.d.geom_xpos[obj]),
                ref.queries.request, result))
            assert np.isfinite(distance)
            distances.append(distance)
        material = min(distances)
        h = float(np.min((vertices @ rotation.T + p - anchor) @ normal))
        point_side = float((p-anchor) @ normal)
        residuals = dict(surface_gap=gap-source_gap, geometry_side=min(h+tau, 0.),
                         signed_recovery=min(material, 0.))
        weights = dict(surface_gap=1024., geometry_side=10000., signed_recovery=1024.)
        return dict(offset_along_source_direction_mm=offset_mm,
            gap_m=gap, geometry_side_m=h, reference_side_m=point_side,
            material_signed_distance_m=material, raw_residuals_m=residuals,
            weighted_residuals={k:np.sqrt(weights[k])*v for k,v in residuals.items()},
            squared_costs={k:weights[k]*v*v for k,v in residuals.items()},
            hard_constraint_margin_at_zero_step_m=h+tau,
            minimum_linearized_normal_gain_needed_m=max(-h-tau, 0.),
            general_collision_slack_needed_at_zero_step_m=max(-material-tau, 0.),
            material_piece=mujoco.mj_id2name(ref.m, mujoco.mjtObj.mjOBJ_GEOM,
                int(ref.material[geom][int(np.argmin(distances))])),
            intended_contact=bool(point_side >= -.001 and gap <= .002 and material >= -.001),
            physical_contact=bool(-.001 <= material <= .002),
            strict_physical_contact=bool(-1e-5 <= material <= .002))

    cases = {name:measure(offset) for name, offset in
             [('saved_correct_contact', 0.), ('separated_probe', 10.),
              ('penetrating_probe', -3.), ('wrong_side_probe', -25.)]}
    # Validate direct shape queries and effective residual weights against solver methods.
    ref.fk(q)
    measured = cases['saved_correct_contact']
    assert abs(ref.gap_query(target, side, finger, False)[0] - measured['gap_m']) < 1e-9
    assert abs(ref.material_query(side, finger, False)[0] - measured['material_signed_distance_m']) < 1e-9
    assert abs(ref.support_side(target, side, finger, False)[0] - measured['geometry_side_m']) < 1e-9
    r, _ = ref.additional(q, target)
    active = np.argwhere(data['contact'][t])
    idx = int(np.flatnonzero((active == [side, finger]).all(1))[0])
    assert np.allclose(r[[2*idx, 2*idx+1, 2*len(active)+idx]],
        [measured['weighted_residuals'][k] for k in ('geometry_side', 'surface_gap', 'signed_recovery')],
        rtol=0, atol=1e-9)
    name = lambda kind, index: mujoco.mj_id2name(ref.m, kind, int(index))
    coverage = sorted(set(name(mujoco.mjtObj.mjOBJ_BODY, ref.m.geom_bodyid[g])
                          for g in ref.pairs[ref.pairs[:,2] == 0, 0]
                          if 'finger' in name(mujoco.mjtObj.mjOBJ_BODY, ref.m.geom_bodyid[g])))
    actual_baseline = OUT / 'variants/baseline/clips' / seq / 'final_v2.npz'
    ref.fk(np.load(actual_baseline)['qpos'][t])
    bp = ref.keypoints(False)[side, 4]
    baseline = dict(gap_m=ref.gap_query(target, side, finger, False)[0],
        geometry_side_m=ref.support_side(target, side, finger, False)[0],
        reference_side_m=float((bp-anchor)@normal),
        material_signed_distance_m=ref.material_query(side, finger, False)[0])
    result = dict(sequence=seq, zero_based_frame=t, time_seconds=t/30.,
        description='Saved hard-condition thumb plus controlled rigid translations of only its collision shape; probes are not full robot poses.',
        source_contact=bool(data['contact'][t, side, finger]),
        source_landmark_distance_m=float(data['source_landmark_distance'][t, side, finger]),
        source_gap_target_m=source_gap, anchor_object_local_m=anchor_local.tolist(),
        direction_object_local=normal_local.tolist(),
        patch_triangle_count=int(end-start),
        patch_axis_min_relative_m=(triangles.reshape(-1,3)-anchor_local).min(0).tolist(),
        patch_axis_max_relative_m=(triangles.reshape(-1,3)-anchor_local).max(0).tolist(),
        patch_area_m2=float(np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],
                                                  triangles[:,2]-triangles[:,0]), axis=1).sum()/2),
        human_skin_triangle_count=int(data['source_skin_triangle_counts'][t, side, finger]),
        robot_tip=dict(body=name(mujoco.mjtObj.mjOBJ_BODY, ref.m.geom_bodyid[geom]),
                       mesh=name(mujoco.mjtObj.mjOBJ_MESH, ref.m.geom_dataid[geom]),
                       support_vertex_count=len(vertices)),
        material_piece_count=len(ref.material[geom]), coal_version=coal.__version__,
        finger_collision_links=coverage, cases=cases, saved_baseline=baseline,
        assertions_passed=True, units='Lengths in metres unless suffix says otherwise; squared objective costs use the numerical coefficients in the solver.',
        input_sha256={str(p):sha(p) for p in (pose_file, actual_baseline,
            OUT/'targets'/f'{seq}.npz', CODE/'explain_frame.py')})
    folder = OUT / 'diagnostics/frame120_contact_terms'
    write(folder/'measurements.json', result)
    print(folder/'measurements.json')
    for label, case in cases.items():
        print(label, 'lengths_mm', [round(case[k]*1000,6) for k in
              ('gap_m','geometry_side_m','material_signed_distance_m','hard_constraint_margin_at_zero_step_m')],
              'costs', case['squared_costs'])


if __name__ == '__main__':
    main()
