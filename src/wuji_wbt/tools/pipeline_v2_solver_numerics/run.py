"""Two full-clip regressions with frozen inputs and the original evaluators.

Launch under a low-priority, CPU-only systemd unit with 200% CPU quota,
MemoryHigh=4G, MemoryMax=5G, and MemorySwapMax=0. Outputs are isolated from
the published signed-contact experiment; no old solver results are rewritten.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
import csv
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
from config import *


def prepare():
    assert not OUT.exists(), 'Never overwrite regression results'
    old = read(PREVIOUS / 'protocol.json')
    assert read(PREVIOUS / 'status.json')['state'] == 'complete'
    for group in ('input_sha256', 'target_sha256'):
        for path, expected in old[group].items():
            assert sha(path) == expected, path
    # The two evaluators and all contact geometry must be unchanged.
    frozen = [BASE_CODE / 'evaluate.py', PILOT_CODE / 'evaluate_one.py',
              BASE_CODE / 'geometry.py', BASE_CODE / 'sensitivity.py']
    for path in frozen:
        assert sha(path) == old['source_sha256'][str(path)], path
    OUT.mkdir(parents=True)
    shutil.copytree(PREVIOUS / 'targets', OUT / 'targets')
    sources = {Path(p) for p in old['source_sha256']} | set(CODE.glob('*.py'))
    sources.add(RET / 'retarget_pipeline/clarabel_step.py')
    current = {str(p): sha(p) for p in sorted(sources)}
    changed = {p: dict(previous=h, current=current[p]) for p, h in old['source_sha256'].items()
               if current[p] != h}
    preserved = {str(PREVIOUS / 'protocol.json'): sha(PREVIOUS / 'protocol.json')}
    for seq in COHORT:
        for filename in ('final_v2.npz', 'run.json', 'diagnostics.json', 'metrics.json'):
            path = PREVIOUS / 'variants/signed/clips' / seq / filename
            preserved[str(path)] = sha(path)
    protocol = dict(version='v2-unscaled-checked-clarabel-regression-1',
        sequences=COHORT, baseline=str(PREVIOUS), baseline_variant='signed',
        numerical_change=dict(tol_gap_abs=1e-6, tol_gap_rel=1e-6, tol_feas=1e-7,
            reduced_tolerances='Installed Clarabel defaults', objective_scale=1.,
            original_unit_feasibility_limit=1e-6,
            fallback='InsufficientProgress/MaxIterations: finite feasible x plus independent dual lower-bound gap. Project z to dual cone, compute g(z)=-.5*(q+A.T*z).T*P^-1*(q+A.T*z)-b.T*z. Require f(x)-g(z) within max(abs_tol,rel_tol*max(1,abs(f),abs(g))).',
            no_post_solve_clipping=True),
        unchanged='All contact/posture weights, damping, collision slack penalty, collision pairs, joint/rate/trust constraints, nonlinear line search, initialization, body/object motions, masks and evaluation definitions.',
        scope='Both complete existing development clips. This tests numerical reliability, not a new contact model.',
        resources=dict(workers=2, cpu_only=True, cpu_quota_percent=200, nice=19,
                       memory_high_gib=4, memory_max_gib=5, memory_swap_max=0),
        source_sha256=current, changes_from_previous_snapshot=changed,
        input_sha256=old['input_sha256'],
        target_sha256={str(p): sha(p) for p in (OUT / 'targets').glob('*.npz')},
        preserved_baseline_sha256=preserved)
    write(OUT / 'protocol.json', protocol)
    for source in sources:
        dest = OUT / 'source_snapshot' / source.relative_to(WORK)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def command(script, args, log):
    start = time.perf_counter()
    with Path(log).open('w') as stream:
        subprocess.run([PY, str(CODE / script), *args], cwd=WORK,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter()-start


def one(seq):
    dest = destination(seq, 'signed'); dest.mkdir(parents=True, exist_ok=True)
    solve_s = command('solver.py', [seq, 'signed'], dest / 'solver.log')
    eval_s = command('run.py', ['evaluate', seq], dest / 'evaluation.log')
    return dict(sequence=seq, solve_process_seconds=solve_s, evaluation_process_seconds=eval_s)


def validate(done):
    from solver import load
    protocol = read(OUT / 'protocol.json')
    for group in ('source_sha256', 'input_sha256', 'target_sha256', 'preserved_baseline_sha256'):
        for path, expected in protocol[group].items():
            assert sha(path) == expected, (group, path)
    rows = []
    for seq in COHORT:
        dest = destination(seq, 'signed')
        previous = PREVIOUS / 'variants/signed/clips' / seq
        Q = np.load(dest / 'final_v2.npz')['qpos']
        old_Q = np.load(previous / 'final_v2.npz')['qpos']
        _, _, _, ref, body, _ = load(seq, 'signed')
        assert Q.shape == old_Q.shape == body.shape and np.isfinite(Q).all()
        fixed = np.setdiff1d(np.arange(ref.m.nq), ref.qadr)
        assert np.array_equal(Q[:, fixed], body[:, fixed])
        assert np.all(Q[:, ref.qadr] >= ref.lo-1e-7)
        assert np.all(Q[:, ref.qadr] <= ref.hi+1e-7)
        steps = [s for r in read(dest / 'iterations.json') for s in r['steps']]
        assert all(s['numerical_objective_scale'] == 1. for s in steps)
        assert all(s['maximum_cone_constraint_violation'] <= 1e-6 for s in steps)
        assert all(s['status'] in ('Solved', 'AlmostSolved') or s['accepted_feasible_fallback'] for s in steps)
        current = read(dest / 'diagnostics.json'); before = read(previous / 'diagnostics.json')
        run = read(dest / 'run.json'); old_run = read(previous / 'run.json')
        assert current['definitions'] == before['definitions']
        assert current['frames'] == before['frames'] == len(Q)
        assert run['settings'] == old_run['settings'] and run['extra'] == old_run['extra']
        row = dict(sequence=seq, frames=len(Q), qp_steps=len(steps),
            qp_status_counts=dict(Counter(s['status'] for s in steps)),
            certified_fallbacks=sum(s['accepted_feasible_fallback'] for s in steps),
            maximum_qp_constraint_violation=max(s['maximum_cone_constraint_violation'] for s in steps),
            frozen_body_object_exact=True, all_objective_scales_one=True,
            active_joint_rms_change_rad=float(np.sqrt(np.mean((Q[:, ref.qadr]-old_Q[:, ref.qadr])**2))),
            active_joint_max_change_rad=float(np.max(abs(Q[:, ref.qadr]-old_Q[:, ref.qadr]))),
            solve_seconds=run['joint_refinement_seconds'])
        metric_names = ['physical_patch_gap_mean_mm', 'object_max_penetration_mm',
            'object_mean_frame_max_penetration_mm', 'self_max_penetration_mm',
            'arm_acceleration_rms_rad_s2', 'intended_patch_touch_recall_skin_contact_pct']
        for metric in metric_names:
            row[metric+'_old'] = before[metric]; row[metric+'_new'] = current[metric]
        for prefix in ('physical_touch_vs_original_source', 'strict_physical_touch_vs_original_source'):
            row[prefix+'_f1_pct_old'] = before[prefix]['f1_pct']
            row[prefix+'_f1_pct_new'] = current[prefix]['f1_pct']
        rows.append(row)
    write(OUT / 'validation.json', dict(passed=True, sequences=rows, timings=done,
        caveat='Relaxed solver stopping can change later nonconvex SQP iterates; trajectories need not be identical. Existing contact thresholds and metrics are unchanged.'))
    with (OUT / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    prepare()
    start = time.perf_counter()
    write(OUT / 'status.json', dict(state='testing', completed=0, total=2))
    command('test_solver.py', [], OUT / 'solver_tests.log')
    done, errors = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(one, seq): seq for seq in COHORT}
        while pending:
            finished, _ = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
            for future in finished:
                seq = pending.pop(future)
                try:
                    done.append(future.result())
                except Exception as exc:
                    errors.append(dict(sequence=seq, error=repr(exc)))
            write(OUT / 'status.json', dict(state='running', completed=len(done), total=2,
                pending=list(pending.values()), errors=errors, elapsed_seconds=time.perf_counter()-start))
    if errors:
        raise RuntimeError(errors)
    validate(done)
    write(OUT / 'status.json', dict(state='complete', completed=2, total=2,
          elapsed_seconds=time.perf_counter()-start, validation_passed=True))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
        module('numerics_frozen_evaluator', BASE_CODE / 'evaluate.py').main(sys.argv[2], 'signed')
    else:
        try:
            main()
        except Exception as exc:
            if OUT.exists():
                status = read(OUT / 'status.json') if (OUT / 'status.json').exists() else {}
                write(OUT / 'status.json', dict(status, state='failed', error=repr(exc)))
            raise
