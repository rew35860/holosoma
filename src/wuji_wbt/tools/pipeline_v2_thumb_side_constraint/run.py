"""Validate a strict side-ownership candidate separately from earlier pilots."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
import shutil
import subprocess
import time
import numpy as np
from config import *


def call(script, args, log):
    with Path(log).open('w') as stream:
        subprocess.run([PY, str(script), *args], cwd=WORK, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def main():
    assert not OUT.exists()
    old = read(BASELINE / 'protocol.json'); seq = COHORT[0]
    for group in ('source_sha256', 'input_sha256', 'target_sha256'):
        for path, expected in old[group].items(): assert sha(path) == expected, path
    assert (BASELINE / 'variants/combined/clips' / seq / 'completed.json').exists()
    OUT.mkdir(parents=True); shutil.copytree(BASELINE / 'targets', OUT / 'targets')
    for variant in ('baseline', 'combined'):
        shutil.copytree(BASELINE / 'variants' / variant / 'clips' / seq, destination(seq, variant))
    sources = {Path(p) for p in old['source_sha256']} | set(CODE.glob('*.py'))
    sources.add(FOLLOWUP_CODE / 'render_cavity.py')
    write(OUT / 'protocol.json', dict(version='right-thumb-hard-side-pilot-1', sequence=seq,
        baseline=str(BASELINE), comparison_variant='combined',
        change='Add the right-thumb support inequality d_min(q)+grad(d_min)*dq >= -0.001 to each QP during existing source-contact frames, without a slack variable. Same whole-tip geometry, source normal, and anchors.',
        unchanged='Combined contact objective and weights, body/object motions, initialization, collision geometry/soft constraints, joint/rate/trust bounds, numerical tolerances, iteration budget, nonlinear merit and line search.',
        evaluation='Original metrics/masks unchanged. Independently measure final nonlinear whole-tip side clearance; a linearized constraint alone is not proof of final feasibility.',
        source_sha256={str(p): sha(p) for p in sources}, input_sha256=old['input_sha256'],
        target_sha256={str(p): sha(p) for p in (OUT / 'targets').glob('*.npz')}))
    for p in sources:
        dest = OUT / 'source_snapshot' / p.relative_to(WORK)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
    start = time.perf_counter(); dest = destination(seq, 'hard'); dest.mkdir(parents=True)
    write(OUT / 'status.json', dict(state='solving'))
    call(CODE / 'solver.py', [seq, 'hard'], dest / 'solver.log')
    call(CODE / 'run.py', ['evaluate'], dest / 'evaluation.log')
    from solver import load, target_at
    audit = module('hard_side_audit', PARENT_CODE / 'audit.py').audit(seq, 'hard')
    arrays = np.load(OUT / 'audit/hard.npz'); mask = arrays['source_contact'][:, 1, 0]
    violation = float(np.max(np.maximum(-arrays['support_side'][:, 1, 0][mask]-.001, 0)))
    valid = violation <= 1e-6
    write(OUT / 'validation.json', dict(nonlinear_side_passed=valid,
        nonlinear_side_violation_m=violation, right_thumb_audit=audit,
        general_metrics=read(dest / 'diagnostics.json'), elapsed_seconds=time.perf_counter()-start))
    write(dest / 'completed.json', dict(state='complete', nonlinear_side_passed=valid))
    write(OUT / 'status.json', dict(state='complete', nonlinear_side_passed=valid,
                                  elapsed_seconds=time.perf_counter()-start))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        module('hard_frozen_evaluator', BASE_CODE / 'evaluate.py').main(COHORT[0], 'hard')
    else:
        try: main()
        except Exception as exc:
            if OUT.exists(): write(OUT / 'status.json', dict(state='failed', error=repr(exc)))
            raise
