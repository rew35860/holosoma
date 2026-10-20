"""Isolated frame-140 diagnostic; never changes production models or evaluations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import cvxpy as cp
import mujoco
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh

WORK = Path(__file__).resolve().parents[4]
RET = WORK/'src/holosoma_retargeting/holosoma_retargeting'
PILOT = WORK.parent/'holosoma/eval_out/ref_candidates/pipeline_v2_pilot_20261018'
SEQ = 'pick_up_and_carry_organizer_medium_while_walking-096'
FRAME = 140
sys.path.insert(0, str(RET/'retarget_pipeline'))
from pipeline_v2 import CoupledRefiner, Settings, TIPS


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def query(model, data, a, b, cutoff):
    ft = np.zeros(6)
    distance = float(mujoco.mj_geomDistance(model, data, a, b, cutoff, ft))
    return dict(cutoff_m=cutoff, signed_distance_m=distance,
                witness_points_world_m=ft.reshape(2, 3).tolist(),
                witness_separation_m=float(np.linalg.norm(ft[:3]-ft[3:])))


def world_vertices(model, data, geom):
    mesh = int(model.geom_dataid[geom])
    start = int(model.mesh_vertadr[mesh]);count = int(model.mesh_vertnum[mesh])
    return (model.mesh_vert[start:start+count].astype(float)
            @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=PILOT/'diagnostics/organizer_frame140_collision')
    args = parser.parse_args();args.output.mkdir(parents=True, exist_ok=True)
    clip = PILOT/'clips'/SEQ;run = read(clip/'run.json')
    code = RET/'retarget_pipeline/pipeline_v2.py'
    assert digest(code) == run['code_sha256'], 'Diagnostic requires the original solver source'
    m = mujoco.MjModel.from_xml_path(run['model'])
    r = CoupledRefiner(m, 'organizer_medium', Settings(**run['settings']))
    Q = np.load(clip/'final_v2.npz')['qpos']
    init = np.load(clip/'finger_initialization.npz')['qpos']
    archived = read(clip/'iterations.json')[FRAME]
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'wjr_right_finger4_link2')
    matches = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body
               and (m.geom_contype[g] or m.geom_conaffinity[g])]
    assert len(matches) == 1
    a = matches[0]
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'organizer_medium_piece_017')
    pair_id = int(np.flatnonzero((r.pairs[:, 0] == a) & (r.pairs[:, 1] == b))[0])
    r.fk(Q[FRAME])
    original = dict(disableflags=int(m.opt.disableflags), enableflags=int(m.opt.enableflags),
                    ccd_iterations=int(m.opt.ccd_iterations), ccd_tolerance=float(m.opt.ccd_tolerance))
    distances = [query(m, r.d, a, b, cutoff) for cutoff in (.001, .01, .025, .03, .04, .05, .1)]
    va, vb = world_vertices(m, r.d, a), world_vertices(m, r.d, b)
    ha, hb = ConvexHull(va), ConvexHull(vb)
    aabb_gap = float(va[:, 0].min()-vb[:, 0].max())
    assert aabb_gap > .028, 'Independent x-coordinate intervals must be disjoint'
    # These are the actual compiled convex collision meshes, independent of mj_geomDistance.
    center = np.mean(np.vstack([va, vb]), axis=0)
    pa, pb = cp.Variable(3), cp.Variable(3)
    constraints = [h.equations[:, :3]@p <= -h.equations[:, 3]-h.equations[:, :3]@center
                   for h, p in ((ha, pa), (hb, pb))]
    problem = cp.Problem(cp.Minimize(cp.sum_squares(pa-pb)), constraints)
    problem.solve(solver='CLARABEL', tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    ca, cb = pa.value+center, pb.value+center
    separation = float(np.linalg.norm(ca-cb));axis = (ca-cb)/separation
    interval_a = [float((va@axis).min()), float((va@axis).max())]
    interval_b = [float((vb@axis).min()), float((vb@axis).max())]
    lower = interval_a[0]-interval_b[1]
    feasibility = max(float((h.equations[:, :3]@p+h.equations[:, 3]).max())
                      for h, p in ((ha, ca), (hb, cb)))
    assert lower > .0339 and abs(separation-lower) < 1e-6 and feasibility < 1e-8
    equations = np.vstack([ha.equations, hb.equations])
    lp = linprog(np.zeros(3), A_ub=equations[:, :3], b_ub=-equations[:, 3],
                 bounds=[(None, None)]*3, method='highs')
    assert lp.status == 2, 'Convex hull intersection must be infeasible'
    independent = dict(closest_point_qp_status=problem.status, separation_m=separation,
                       certified_projected_gap_m=lower, certificate_gap_m=separation-lower,
                       closest_point_halfspace_max_residual_m=feasibility,
                       world_x_aabb_gap_m=aabb_gap, separating_axis_world=axis.tolist(),
                       projected_interval_robot_m=interval_a, projected_interval_object_m=interval_b,
                       closest_points_world_m=[ca.tolist(), cb.tolist()],
                       intersection_lp_status=int(lp.status), intersect=False)

    sweeps = []
    for iterations in (10, 35, 100, 500):
        m.opt.ccd_iterations = iterations
        sweeps.append(dict(iterations=iterations, query=query(m, r.d, a, b, .05)))
    m.opt.ccd_iterations = original['ccd_iterations']
    for tolerance in (1e-4, 1e-6, 1e-8, 1e-10):
        m.opt.ccd_tolerance = tolerance
        sweeps.append(dict(tolerance=tolerance, query=query(m, r.d, a, b, .05)))
    m.opt.ccd_tolerance = original['ccd_tolerance']
    m.opt.disableflags = original['disableflags'] | int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    legacy = query(m, r.d, a, b, .05)
    m.opt.disableflags = original['disableflags']
    neighbours = []
    for t in range(FRAME-2, FRAME+3):
        r.fk(Q[t]);neighbours.append(dict(frame=t, query=query(m, r.d, a, b, .05)))

    z = np.load(run['canonical'])
    human, pose, O = z['hand_kp_scaled'][FRAME], z['object_pose_mujoco'][FRAME], z['object_points_scaled']
    R = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    local = (human[:, TIPS]-pose[:3])@R
    mesh = trimesh.load(RET/'models/organizer_medium/organizer_medium.obj', force='mesh')
    mesh.apply_scale(float(z['scale']))
    near, source_distances, _ = trimesh.proximity.ProximityQuery(mesh).on_surface(local.reshape(-1, 3))
    contact = (source_distances < .03).reshape(2, 5)
    surface = near.reshape(2, 5, 3)@R.T+pose[:3]
    target = r.frame_target(human, O, pose, contact, surface)
    previous = Q[FRAME-1];q = init[FRAME].copy();q[r.qadr] = previous[r.qadr]
    assert np.array_equal(q, Q[FRAME])
    residual, J = r.residuals(q, target, init[FRAME], init[FRAME], previous)
    dist, G, ids = r.collisions(q)
    sphere_lower = float(np.linalg.norm(r.d.geom_xpos[a]-r.d.geom_xpos[b])
                         - m.geom_rbound[a]-m.geom_rbound[b])
    assert sphere_lower < r.cfg.collision_detect and pair_id not in ids
    step, info = r.subproblem(q, residual, J, dist, G, previous)
    penalty = .5*r.cfg.collision_weight*np.sum(np.maximum(-dist-r.cfg.collision_tolerance, 0)**2)
    merit = float(residual@residual+penalty)
    for key, value in [('merit', merit), ('step_norm', float(np.linalg.norm(step))),
                       ('slack_max_m', info['slack_max_m'])]:
        assert abs(value-archived['steps'][0][key]) < 1e-9
    trials = []
    for alpha in (1., .5, .25, .125):
        trial = q.copy();trial[r.qadr] += alpha*step
        rr, _ = r.residuals(trial, target, init[FRAME], init[FRAME], previous)
        dd, _, ii = r.collisions(trial, derivatives=False)
        value = float(rr@rr + .5*r.cfg.collision_weight*np.sum(np.maximum(-dd-r.cfg.collision_tolerance, 0)**2))
        trials.append(dict(alpha=alpha, merit=value, accepted=value <= merit+1e-9,
                           pair_has_constraint=bool(pair_id in ii), query=query(m, r.d, a, b, .05)))
    assert not any(t['accepted'] for t in trials)
    replay = dict(archived=archived, replay_status=info, replay_merit=merit,
                  replay_step_norm=float(np.linalg.norm(step)), trials=trials,
                  output_equals_previous_optimized_joint_values=bool(np.array_equal(Q[FRAME, r.qadr], previous[r.qadr])),
                  registered_pair_id=pair_id, passes_sphere_filter=True, sphere_lower_bound_m=sphere_lower,
                  pair_has_constraint=False, omitted_reason='25mm query returns cutoff, triggering dist >= collision_detect skip',
                  pair_uses_slack=False, replay_matches_archived_to_1e_9=True)
    def describe(g):
        return dict(id=g, geom=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g),
                    body=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])),
                    mesh=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[g])),
                    contype=int(m.geom_contype[g]), conaffinity=int(m.geom_conaffinity[g]))
    files = [Path(__file__), code, Path(run['model']), Path(run['canonical']),
             clip/'run.json', clip/'iterations.json', clip/'final_v2.npz', clip/'finger_initialization.npz',
             clip/'metrics.json', clip/'per_frame_metrics.npz']
    result = dict(sequence=SEQ, frame_zero_based=FRAME, time_s=FRAME/run['settings']['fps'],
                  conclusion='The reported -34.007924mm distance is a false positive; the convex geometries are separated by +34.007924mm.',
                  mujoco_version=mujoco.__version__, original_options=original, geometry_a=describe(a), geometry_b=describe(b),
                  cutoff_sweep=distances, independent_geometry=independent, ccd_setting_sweeps=sweeps,
                  legacy_ccd_comparison=legacy, neighbouring_frames=neighbours, frame_replay=replay,
                  original_evaluations_unchanged=True, production_settings_unchanged=True,
                  limitation='Only this flagged geometry pair and frame are independently certified. Other penetration measurements are not validated here.',
                  source_sha256={str(p): digest(p) for p in files})
    (args.output/'diagnosis.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(args.output/'geometry_certificate.npz', robot_vertices_world=va,
                        object_vertices_world=vb, separating_axis=axis, closest_robot=ca, closest_object=cb)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(12, 5.4));ax = fig.add_subplot(121, projection='3d')
    origin = cb
    for vertices, hull, color in ((vb, hb, '#438ec6'), (va, ha, '#e18c36')):
        points = (vertices-origin)*1000
        ax.add_collection3d(Poly3DCollection(points[hull.simplices], alpha=.40, facecolor=color,
                                             edgecolor=color, linewidth=.18))
    segment = (np.vstack([cb, ca])-origin)*1000
    ax.plot(*segment.T, color='black', linewidth=2, marker='o')
    allv = (np.vstack([va, vb])-origin)*1000;lo, hi = allv.min(0), allv.max(0)
    ax.set(xlim=(lo[0]-5, hi[0]+5), ylim=(lo[1]-5, hi[1]+5), zlim=(lo[2]-5, hi[2]+5),
           xlabel='World x offset (mm)', ylabel='World y offset (mm)', zlabel='World z offset (mm)',
           title='Actual convex collision geometry')
    ax.set_box_aspect(hi-lo);ax.view_init(elev=25, azim=-65)
    ax2 = fig.add_subplot(122)
    vertical = np.array([0., 0., 1.]);vertical -= axis*(vertical@axis);vertical /= np.linalg.norm(vertical)
    for v, color, label in ((vb, '#438ec6', 'Organizer piece 017'), (va, '#e18c36', 'Right ring finger link 2')):
        coords = np.c_[(v-origin)@axis, (v-origin)@vertical]*1000;outline = ConvexHull(coords)
        ax2.fill(coords[outline.vertices, 0], coords[outline.vertices, 1], color=color, alpha=.5, label=label)
    ax2.plot([0, separation*1000], [0, (ca-cb)@vertical*1000], 'ko-', linewidth=2)
    ax2.axvline(0, color='grey', linestyle='--', linewidth=.8);ax2.axvline(lower*1000, color='grey', linestyle='--', linewidth=.8)
    ax2.text(separation*500, 9, f'{separation*1000:.3f} mm GAP', ha='center', weight='bold')
    ax2.set(xlabel='Projection along separating axis (mm)', ylabel='Perpendicular projection (mm)',
            title='Disjoint projected intervals prove separation');ax2.set_aspect('equal');ax2.legend(fontsize=8, loc='lower right')
    fig.suptitle('Organizer frame 140 (4.67 s): reported penetration is a distance-query false positive', fontsize=12)
    fig.tight_layout();fig.savefig(args.output/'pair_geometry.png', dpi=170);plt.close(fig)
    print(json.dumps(dict(output=str(args.output), geometry_a=describe(a), geometry_b=describe(b),
                          independent_geometry=independent, replay_matches_archived=True,
                          constraint_for_pair_present=False), indent=2))


if __name__ == '__main__':
    main()
