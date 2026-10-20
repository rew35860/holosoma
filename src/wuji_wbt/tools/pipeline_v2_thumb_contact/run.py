"""Isolated two-worker CPU contact-side pilot, preserving previous results."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from config import *


def command(script, args, log):
    start = time.perf_counter()
    with Path(log).open('w') as f:
        subprocess.run([PY, str(CODE / script), *args], cwd=WORK, stdout=f,
                       stderr=subprocess.STDOUT, check=True)
    return time.perf_counter()-start


def prepare():
    assert not OUT.exists(), 'Never overwrite previous pilot outputs'
    old = read(BASELINE / 'protocol.json')
    assert read(BASELINE / 'validation.json')['passed']
    for group in ('source_sha256', 'input_sha256', 'target_sha256'):
        for path, expected in old[group].items():
            assert sha(path) == expected, path
    OUT.mkdir(parents=True)
    shutil.copytree(BASELINE / 'targets', OUT / 'targets')
    seq = COHORT[0]
    shutil.copytree(BASELINE / 'variants/signed/clips' / seq, destination(seq, 'baseline'))
    write(destination(seq, 'baseline') / 'completed.json', dict(reused=True, source=str(BASELINE)))
    sources = {Path(p) for p in old['source_sha256']} | set(CODE.glob('*.py'))
    write(OUT / 'protocol.json', dict(version='organizer-contact-side-pilot-1', sequence=seq,
        baseline=str(BASELINE), numerical_policy='Same updated unscaled solver for every condition',
        variants=dict(baseline='Reused signed-contact result with explicit numerical tolerances',
            priority='Only multiply existing direction, regional gap and signed-recovery squared costs by 4 for thumb/index on both hands. Human fingertip tracking and IM unchanged.',
            support='Only replace direction residual min(0,n dot (p_reference-anchor)) with min(0,min_tip_vertex n dot (v-anchor)+0.001). Same coefficient 2500; same source directions and regional patches.'),
        unchanged='Initialization, contact masks/targets, human imitation terms, body/object trajectories, collision constraints/pairs/slack, joint bounds and solver/line-search settings.',
        evaluation='Historical evaluator and supplemental contact metrics unchanged. Additional per-digit side/geometry/patch checks are reported separately.',
        caveat='An oriented local plane encodes intended side, not pad orientation or grasp forces. It can be too restrictive near edges; evaluate before adopting. This is a development test, not evidence of physical grasp stability.',
        resources=dict(cpu_only=True, workers=2, cpu_quota_percent=200, memory_max_gib=5),
        source_sha256={str(p): sha(p) for p in sources},
        input_sha256=old['input_sha256'],
        target_sha256={str(p): sha(p) for p in (OUT / 'targets').glob('*.npz')},
        preserved_baseline_sha256={str(BASELINE / 'variants/signed/clips' / seq / 'final_v2.npz'):
                                  sha(BASELINE / 'variants/signed/clips' / seq / 'final_v2.npz')}))
    for p in sources:
        dest = OUT / 'source_snapshot' / p.relative_to(WORK)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)


def one(variant):
    seq = COHORT[0]; dest = destination(seq, variant); dest.mkdir(parents=True, exist_ok=True)
    solve = command('solver.py', [seq, variant], dest / 'solver.log')
    evaluate = command('run.py', ['evaluate', seq, variant], dest / 'evaluation.log')
    audit = command('audit.py', [seq, variant], dest / 'audit.log')
    result = dict(sequence=seq, variant=variant, state='complete', solve_process_s=solve,
                  evaluate_process_s=evaluate, audit_process_s=audit)
    write(dest / 'completed.json', result)
    return result


def main():
    prepare(); start = time.perf_counter()
    write(OUT / 'status.json', dict(state='diagnosing', completed=1, total=3))
    command('audit.py', [COHORT[0], 'baseline'], OUT / 'baseline_audit.log')
    done, errors = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(one, v): v for v in ('priority', 'support')}
        while pending:
            finished, _ = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
            for future in finished:
                variant = pending.pop(future)
                try: done.append(future.result())
                except Exception as exc: errors.append(dict(variant=variant, error=repr(exc)))
            write(OUT / 'status.json', dict(state='running', completed=1+len(done), total=3,
                pending=list(pending.values()), errors=errors, elapsed_seconds=time.perf_counter()-start))
    if errors: raise RuntimeError(errors)
    protocol = read(OUT / 'protocol.json')
    for group in ('source_sha256', 'input_sha256', 'target_sha256', 'preserved_baseline_sha256'):
        for path, expected in protocol[group].items(): assert sha(path) == expected, path
    write(OUT / 'status.json', dict(state='rendering', completed=3, total=3))
    command('render.py', [COHORT[0]], OUT / 'render.log')
    command('report.py', [], OUT / 'report.log')
    write(OUT / 'status.json', dict(state='complete', completed=3, total=3,
        elapsed_seconds=time.perf_counter()-start, validation_passed=True, timings=done))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
        module('thumb_frozen_evaluator', BASE_CODE / 'evaluate.py').main(*sys.argv[2:])
    else:
        try: main()
        except Exception as exc:
            if OUT.exists(): write(OUT / 'status.json', dict(state='failed', error=repr(exc)))
            raise
