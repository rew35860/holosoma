"""Check the exact removed residual and final invariant/evaluation/media outputs."""
import numpy as np
from config import *
from solver import load, target_at


def hashes():
    protocol = read(OUT / 'protocol.json')
    for group in ('input_sha256', 'source_sha256', 'target_sha256'):
        for path, expected in protocol[group].items():
            assert sha(path) == expected, path
    for seq, entries in protocol['reused_baseline_sha256'].items():
        for relative, expected in entries.items():
            assert sha(destination(seq, 'both') / relative) == expected, (seq, relative)


def preflight():
    hashes()
    checks = []
    for seq in COHORT:
        _, z, data, control, body, init = load(seq, 'both')
        _, _, _, surface, _, _ = load(seq, 'surface')
        a, b = vars(control.cfg).copy(), vars(surface.cfg).copy()
        assert {k for k in a if a[k] != b[k]} == {'contact_weight'}
        assert a['contact_weight'] == 100. and b['contact_weight'] == 0.
        assert np.array_equal(control.pairs, surface.pairs)
        Q = np.load(destination(seq, 'both') / 'final_v2.npz')['qpos']
        frames = np.flatnonzero(data['contact'].any(axis=(1, 2)))
        for t in [int(frames[0]), int(frames[len(frames)//2]), int(frames[-1])]:
            tc, ts = target_at(control, z, data, t), target_at(surface, z, data, t)
            q = Q[t].copy()
            previous = Q[t-1] if t else None
            rc, Jc = control.residuals(q, tc, body[t], init[t], previous)
            rs, Js = surface.residuals(q, ts, body[t], init[t], previous)
            control.fk(q)
            p, jac = control.keypoints()
            expected_cost = 0.
            expected_grad = np.zeros(control.n)
            for side in range(2):
                active = data['contact'][t, side]
                tips = np.array([4, 8, 12, 16, 20])[active]
                delta = p[side, tips] - tc['surface'][side, active]
                expected_cost += 100. * np.sum(delta**2)
                expected_grad += 200. * np.einsum('ij,ijk->k', delta, jac[side, tips])
            cost_error = float(abs((rc @ rc - rs @ rs) - expected_cost))
            grad_error = float(np.max(abs(2*(Jc.T @ rc - Js.T @ rs) - expected_grad)))
            assert cost_error < 1e-8 and grad_error < 1e-8, (seq, t, cost_error, grad_error)
            for old, new in zip(control.additional(q, tc), surface.additional(q, ts)):
                assert np.array_equal(old, new)
            # Every changed row must be precisely zero in the new residual/Jacobian.
            changed = np.any(Jc != Js, axis=1) | (rc != rs)
            assert np.all(rs[changed] == 0.) and np.all(Js[changed] == 0.)
            checks.append(dict(sequence=seq, frame=t, removed_cost_error=cost_error,
                               removed_gradient_error=grad_error, other_residuals_exact=True,
                               collision_pairs_exact=True))
    write(OUT / 'preflight.json', dict(passed=True, checks=checks))


def validate():
    from PIL import Image
    import imageio_ffmpeg
    hashes()
    assert read(OUT / 'preflight.json')['passed']
    videos = []
    for seq in COHORT:
        control = read(destination(seq, 'both') / 'run.json')
        new = read(destination(seq, 'surface') / 'run.json')
        assert {k for k in control['settings'] if control['settings'][k] != new['settings'][k]} == {'contact_weight'}
        assert new['settings']['contact_weight'] == 0.
        assert new['extra'] == control['extra']
        assert new['collision_pair_counts'] == control['collision_pair_counts']
        for variant in REMOVALS:
            dest = destination(seq, variant)
            Q = np.load(dest / 'final_v2.npz')['qpos']
            assert np.isfinite(Q).all()
            metrics = read(dest / 'metrics.json')
            assert metrics['unchanged_source_contact_mask'] and metrics['unchanged_body_exact']
            assert read(dest / 'diagnostics.json')['frozen_body_object_exact']
        for suffix in ('', '_hands', '_posture'):
            with Image.open(OUT / 'renders' / f'{seq}{suffix}.png') as im:
                im.verify()
        path = OUT / 'renders' / f'{seq}.mp4'
        decoder = imageio_ffmpeg.read_frames(str(path), input_params=['-threads', '1'], output_params=['-threads', '1'])
        meta = next(decoder)
        count = sum(1 for _ in decoder)
        assert count == (len(Q)+1)//2 and meta['size'] == (1440, 540), (seq, count, meta)
        videos.append(dict(sequence=seq, video_frames=count, size=meta['size'], sha256=sha(path)))
    write(OUT / 'validation.json', dict(passed=True, baseline_unchanged=True,
          source_targets_and_evaluation_unchanged=True, sole_setting_change='contact_weight: 100 -> 0', videos=videos))
