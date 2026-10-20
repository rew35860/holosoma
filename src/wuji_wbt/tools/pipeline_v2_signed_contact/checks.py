"""Independent convex intersection, analytic sign and residual gradient checks."""
import warnings
import numpy as np
import coal
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from config import *
from solver import load, target_at, CoupledRefiner


def hashes():
    protocol = read(OUT / 'protocol.json')
    for group in ('input_sha256', 'source_sha256', 'target_sha256'):
        for path, expected in protocol[group].items():
            assert sha(path) == expected, path
    for seq, entries in protocol['reused_baseline_sha256'].items():
        for relative, expected in entries.items():
            assert sha(destination(seq, 'surface') / relative) == expected, (seq, relative)


def convex_overlap_radius(ref, pair):
    equations = []
    for geom in pair:
        shape = ref.queries.shapes[geom]
        local = np.asarray([shape.point(i) for i in range(shape.num_points)])
        world = local @ ref.d.geom_xmat[geom].reshape(3, 3).T + ref.d.geom_xpos[geom]
        equations.append(ConvexHull(world).equations)
    eq = np.vstack(equations)
    # Largest signed inscribed-ball radius of the halfspace intersection.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected')
        result = linprog([0., 0., 0., -1.], A_ub=np.c_[eq[:, :3], np.ones(len(eq))],
                         b_ub=-eq[:, 3], bounds=[(None, None)]*4,
                         method='highs', options={'threads': 1})
    assert result.success, result.message
    return float(result.x[3])


def preflight():
    hashes()
    analytic = []
    for z, expected in ((.02, .01), (.01, 0.), (.008, -.002)):
        req = coal.DistanceRequest(True, 0., 0.); req.enable_signed_distance = True
        req.gjk_tolerance = 1e-9; req.epa_tolerance = 1e-9
        result = coal.DistanceResult()
        measured = coal.distance(coal.Sphere(.005), coal.Transform3s(np.eye(3), np.array([0., 0., z])),
                                 coal.Box(.04, .04, .01), coal.Transform3s(), req, result)
        assert abs(measured-expected) < 1e-6, (z, measured, expected)
        analytic.append(dict(sphere_center_z=z, expected_m=expected, measured_m=float(measured)))
    checks, independent = [], []
    rng = np.random.default_rng(20261019)
    for seq in COHORT:
        _, z, data, ref, body, init = load(seq, 'signed')
        _, _, _, baseline, _, _ = load(seq, 'surface')
        assert vars(ref.cfg) == vars(baseline.cfg)
        assert np.array_equal(ref.pairs, baseline.pairs)
        Q = np.load(destination(seq, 'surface') / 'final_v2.npz')['qpos']
        active = np.flatnonzero(data['contact'].any(axis=(1, 2)))
        for t in [int(active[0]), int(active[len(active)//2]), int(active[-1])]:
            target = target_at(ref, z, data, t)
            bt = target_at(baseline, z, data, t)
            q = Q[t].copy(); previous = Q[t-1] if t else None
            r, J = ref.residuals(q, target, body[t], init[t], previous)
            rb, Jb = baseline.residuals(q, bt, body[t], init[t], previous)
            extra, G = ref.recovery(q, target)
            assert np.array_equal(r[:len(rb)], rb) and np.array_equal(J[:len(rb)], Jb)
            assert np.array_equal(r[len(rb):], extra) and np.array_equal(J[len(rb):], G)
            # Confirm the normal (unscaled) solve matches the original V2 path.
            baseline.fk(q)
            distances, gradients, _ = baseline.collisions(q)
            original_step, _ = CoupledRefiner.subproblem(baseline, q, rb, Jb, distances, gradients, previous)
            current_step, numerical_info = baseline.subproblem(q, rb, Jb, distances, gradients, previous)
            assert numerical_info['numerical_objective_scale'] == 1.
            assert np.max(np.abs(original_step-current_step)) < 1e-10
            errors = []
            for columns in (np.concatenate(ref.arm_columns), np.concatenate(ref.finger_columns)):
                for _ in range(3):
                    v = np.zeros(ref.n); v[columns] = rng.normal(size=len(columns)); v /= np.linalg.norm(v)
                    h = 1e-6; qp = q.copy(); qm = q.copy()
                    qp[ref.qadr] += h*v; qm[ref.qadr] -= h*v
                    rp, _ = ref.recovery(qp, target); rm, _ = ref.recovery(qm, target)
                    errors.append(float(np.max(np.abs((rp-rm)/(2*h)-G@v))))
            assert max(errors) < .005, (seq, t, errors)
            ref.fk(q)
            for side, finger in np.argwhere(data['contact'][t]):
                d, _, pair = ref.material_query(int(side), int(finger))
                brute = min(ref.queries.query(ref.d, pair[0], int(obj))[0] for obj in ref.material[pair[0]])
                assert abs(d-brute) < 1e-9, (seq, t, pair, d, brute)
                radius = convex_overlap_radius(ref, pair)
                if d < -1e-5:
                    assert radius > 1e-8, (seq, t, pair, d, radius)
                elif d > 1e-5:
                    assert radius < -1e-8, (seq, t, pair, d, radius)
                independent.append(dict(sequence=seq, frame=t, side=int(side), finger=int(finger),
                                        pair=list(pair), signed_distance_m=d, independent_intersection_radius_m=radius))
            checks.append(dict(sequence=seq, frame=t, maximum_gradient_error=max(errors),
                               other_residuals_exact=True, collision_pairs_exact=True))
    write(OUT / 'preflight.json', dict(passed=True, analytic_sign_checks=analytic,
          gradient_checks=checks, independent_convex_intersections=independent))


def validate():
    from PIL import Image
    import imageio_ffmpeg
    hashes()
    assert read(OUT / 'preflight.json')['passed']
    videos = []
    for seq in COHORT:
        old = read(destination(seq, 'surface') / 'run.json')
        new = read(destination(seq, 'signed') / 'run.json')
        assert old['settings'] == new['settings'] and old['extra'] == new['extra']
        assert old['collision_pair_counts'] == new['collision_pair_counts']
        assert new['signed_contact']['enabled'] and new['signed_contact']['weight'] == SIGNED_WEIGHT
        for variant in REMOVALS:
            dest = destination(seq, variant)
            Q = np.load(dest / 'final_v2.npz')['qpos']; assert np.isfinite(Q).all()
            metrics = read(dest / 'metrics.json')
            assert metrics['unchanged_source_contact_mask'] and metrics['unchanged_body_exact']
            assert read(dest / 'diagnostics.json')['frozen_body_object_exact']
        for suffix in ('', '_hands', '_posture'):
            with Image.open(OUT / 'renders' / f'{seq}{suffix}.png') as im:
                im.verify()
        path = OUT / 'renders' / f'{seq}.mp4'
        decoder = imageio_ffmpeg.read_frames(str(path), input_params=['-threads', '1'], output_params=['-threads', '1'])
        meta = next(decoder); count = sum(1 for _ in decoder)
        assert count == (len(Q)+1)//2 and meta['size'] == (1440, 540), (seq, count, meta)
        videos.append(dict(sequence=seq, video_frames=count, size=meta['size'], sha256=sha(path)))
    write(OUT / 'validation.json', dict(passed=True, baseline_unchanged=True,
          source_targets_and_evaluation_unchanged=True, original_settings_exact=True, videos=videos))
