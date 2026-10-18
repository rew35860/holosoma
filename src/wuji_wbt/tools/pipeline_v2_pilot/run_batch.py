"""Resource-limited matched v2 pilot, preserving original reference data."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parents[4]
ROOT = WORK.parent
RET = WORK/'src/holosoma_retargeting/holosoma_retargeting'
BATCH = ROOT/'holosoma/eval_out/ref_candidates/batch_20261011_current_120'
OUT = ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_pilot_20261018'
CODE = Path(__file__).resolve().parent
PY = '/home/natcha/miniconda3/envs/omniretarget/bin/python'

def read(p):
    return json.loads(p.read_text())

def write(p, d):
    tmp = p.with_suffix('.tmp');tmp.write_text(json.dumps(d, indent=2)+'\n');tmp.replace(p)

def run(args, logfile):
    start = time.perf_counter()
    with logfile.open('a') as log:
        log.write('COMMAND '+repr(args)+'\n');log.flush()
        subprocess.run(args, cwd=RET, check=True, stdout=log, stderr=subprocess.STDOUT)
    return time.perf_counter()-start

def one(seq):
    job = read(BATCH/'jobs'/f'{seq}.json');obj = job['object'];clip = BATCH/'clips'/seq
    dest = OUT/'clips'/seq;dest.mkdir(parents=True, exist_ok=True)
    if (dest/'completed.json').exists():
        return read(dest/'completed.json')
    body = clip/'part1'/f'{seq}_stageA.npz'
    canonical = clip/'part0'/f'{seq}_canonical.npz'
    model = Path(job['model_xml'])
    bodymodel = clip/'models'/seq/'stageA'/f'g1_29dof_wuji_welded_w_{obj}.xml'
    # Rerun Omni body in parallel for measured timing and deterministic verification.
    # Joint refinement uses the identical archived body for a paired comparison.
    bodyout = dest/'body_rerun';(bodyout/'part0').mkdir(parents=True, exist_ok=True)
    shutil.copy2(canonical, bodyout/'part0'/canonical.name)
    modelout = bodyout/'models'/seq/'stageA'/bodymodel.name
    modelout.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bodymodel, modelout)
    bodyargs = [PY, str(RET/'retarget_pipeline/pipeline.py'), '--task-name', seq,
        '--object-name', obj, '--stage', 'part1', '--data-path', str(BATCH/'inputs'),
        '--save-dir', str(bodyout), '--pen-tolerance', str(job.get('body_pen_tolerance', .001))]
    args = [PY, str(RET/'retarget_pipeline/pipeline_v2.py'), '--canonical', str(canonical),
        '--body', str(body), '--body-model', str(bodymodel), '--model', str(model),
        '--object', obj, '--output', str(dest)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, bodyargs, dest/'body.log'), pool.submit(run, args, dest/'refinement.log')]
        body_seconds, refinement_seconds = [f.result() for f in futures]
    eval_seconds = run([PY, str(CODE/'evaluate_one.py'), seq], dest/'evaluation.log')
    export_seconds = run([PY, str(WORK/'src/wuji_wbt/motion/convert_wuji.py'),
        str(dest/'final_v2.npz'), str(model), str(dest/'v2_wuji_50hz.npz'), '30', '50'], dest/'export.log')
    import numpy as np
    a = np.load(body)['qpos'];b = np.load(bodyout/'part1'/body.name)['qpos']
    difference = float(np.max(np.abs(a-b)))
    assert a.shape == b.shape
    # The paired run always uses the exact archived Stage-A coordinates even if solver
    # platform details cause a nonzero repeated-solve difference; report rather than hide it.
    result = dict(sequence=seq, status='complete', measured_body_seconds=body_seconds,
        measured_refinement_process_seconds=refinement_seconds, measured_export_seconds=export_seconds,
        total_sequential_retarget_export_seconds=body_seconds+refinement_seconds+export_seconds,
        evaluation_seconds=eval_seconds, repeated_body_max_difference=difference,
        paired_refinement_uses_exact_archived_body=True,
        baseline_stage_seconds={k: v['wall_seconds'] for k, v in job['stages'].items()},
        final_sha256=hashlib.sha256((dest/'final_v2.npz').read_bytes()).hexdigest())
    write(dest/'completed.json', result)
    return result

def main():
    start = time.perf_counter();cohort = read(OUT/'cohort.json')['sequences']
    done, errors = [], []
    for seq in cohort:
        write(OUT/'status.json', dict(state='running', current=seq, completed=len(done), total=len(cohort), errors=errors))
        try:
            done.append(one(seq))
        except Exception as exc:
            errors.append(dict(sequence=seq, error=repr(exc)))
            write(OUT/'status.json', dict(state='failed', current=seq, completed=len(done), total=len(cohort), errors=errors))
            raise
        write(OUT/'completed.json', done)
        print('COMPLETE', seq, flush=True)
    write(OUT/'status.json', dict(state='complete', completed=len(done), total=len(cohort),
        errors=errors, batch_wall_seconds=time.perf_counter()-start))

if __name__ == '__main__':
    main()
