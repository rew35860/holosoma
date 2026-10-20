"""Persistent, bounded CPU batch. Only objective weights differ across variants."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import os
import random
import subprocess
import time
from config import *

def command(args, log):
    start = time.perf_counter()
    with log.open('a') as f:
        f.write('COMMAND '+repr(args)+'\n');f.flush()
        subprocess.run(args, cwd=RET, check=True, stdout=f, stderr=subprocess.STDOUT)
    return time.perf_counter()-start

def one(variant, seq, manifest):
    dest = OUT/'variants'/variant/'clips'/seq;dest.mkdir(parents=True, exist_ok=True)
    completed = dest/'completed.json'
    if completed.exists():
        return read(completed)
    assert sha(SOLVER) == manifest['solver_sha256'], 'Solver changed during experiment'
    for f, expected in manifest['input_sha256'][seq].items():
        assert sha(f) == expected, f'Input changed: {f}'
    job = inputs(seq);conf = OUT/'settings'/f'{variant}.json'
    assert read(conf) == settings(variant)
    solve = command([PY, str(SOLVER), '--canonical', str(job['canonical']),
        '--body', str(job['body']), '--body-model', str(job['body_model']),
        '--model', str(job['model']), '--object', job['object'], '--settings', str(conf),
        '--output', str(dest)], dest/'refinement.log')
    evaluation = command([PY, str(CODE/'evaluate.py'), variant, seq], dest/'evaluation.log')
    run = read(dest/'run.json');checks = read(dest/'checks.json')
    iterations = read(dest/'iterations.json')
    result = dict(variant=variant, sequence=seq, state='complete', frames=run['frames'],
        measured_refinement_process_seconds=solve, evaluation_process_seconds=evaluation,
        preparation_seconds=run['preparation_seconds'],
        finger_initialization_seconds=run['finger_initialization_seconds'],
        joint_refinement_seconds=run['joint_refinement_seconds'],
        total_subproblems=run['total_subproblems'],
        clarabel_solver_seconds=sum(s['solver_seconds'] for r in iterations for s in r['steps']),
        line_search_stopped_frames=sum(r['linesearch_stopped'] for r in iterations),
        checks_passed=checks['passed'], final_sha256=sha(dest/'final_v2.npz'))
    write(completed, result)
    print('COMPLETE', variant, seq, f'{solve:.1f}s', flush=True)
    return result

def report():
    subprocess.run([PY, str(CODE/'report.py')], cwd=CODE, check=True)

def main(workers):
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_file = OUT/'protocol.json'
    if not protocol_file.exists():
        manifest = dict(solver_sha256=sha(SOLVER), input_sha256={seq:{str(p):sha(p) for key,p in inputs(seq).items()
                         if key != 'object'} for seq in COHORT},
                        evaluator_sha256=sha(PILOT_CODE.parent/'ref_candidates_metrics/evaluate.py'),
                        adapter_sha256=sha(PILOT_CODE/'evaluate_one.py'))
        write(OUT/'input_manifest.json', manifest)
        for variant in REMOVALS:
            write(OUT/'settings'/f'{variant}.json', settings(variant))
        write(protocol_file, dict(version='v2-contact-only-arm-guidance-1.0', sequences=COHORT,
            variants={v:dict(label=LABELS[v], changed_settings=REMOVALS[v]) for v in REMOVALS},
            fixed_settings=BASE_SETTINGS, predeclared_before_results=True,
            reference_pilot=str(PILOT), branch='feature/wuji-pipeline-v2',
            scope='Same six complete development clips; 3 variants, 18 newly measured refinement runs. No V1 or body stage rerun.',
            only_changed='Omni arm prior during per-hand source contact: 0.002 versus 0.0005 versus 0. All non-contact weights stay 0.05; all hand objectives unchanged.',
            retained='Quick six-iteration human finger initialization, interaction mesh (including original tip emphasis), main bone term, temporal term, damping, joint limits, rate bound, trust region and all collisions.',
            contacts='Same source-only 30 mm mask and nearest-surface points, computed even when their objective weights are zero.',
            evaluation='Exact frozen pilot evaluator, including its penetration reaggregation; no changed metric definitions or thresholds.',
            evaluation_protocol=read(PILOT/'protocol.json'),
            timing='All variants including full V2 measured anew. Report preparation/init/joint solve/subprocess times. Body reuse, evaluation and visualization excluded from refinement timing; no tracker resampling/export in this study.',
            zero_weights='Same solver code executes zero-weight residual construction too. This isolates objective contribution; times do not estimate a code-pruned implementation.',
            ordering='Seeded shuffled clip order, rotated variant order per clip, then seeded shuffled jobs; seed 20261018.',
            resources=dict(cpu_only=True, workers=workers, cpu_quota_percent=200, memory_high_gib=4,
                           memory_max_gib=5, memory_swap_max=0, nice=19),
            limitations=['Development cohort, not held-out validation.', 'Single timed run per variant/clip; concurrent load varies.',
                         'Relaxing a reference prior does not create an explicit robot comfort objective.', 'Native MuJoCo 3.10 distance sign artifact is known for organizer frame 140; frozen metrics are retained and flagged, not silently corrected.',
                         'Kinematic contact metrics do not establish dynamic grasp stability.']))
    manifest = read(OUT/'input_manifest.json')
    assert manifest['solver_sha256'] == sha(SOLVER)
    assert manifest['evaluator_sha256'] == sha(PILOT_CODE.parent/'ref_candidates_metrics/evaluate.py')
    rng = random.Random(20261018);seqs = COHORT.copy();rng.shuffle(seqs)
    jobs = [];variants = list(REMOVALS)
    for i, seq in enumerate(seqs):
        jobs.extend((v, seq) for v in variants[i:]+variants[:i])
    rng.shuffle(jobs)
    write(OUT/'job_order.json', jobs)
    start = time.perf_counter();done=[];errors=[]
    def status(state, pending):
        write(OUT/'status.json', dict(state=state, completed=len(done), total=len(jobs), workers=workers,
            elapsed_seconds=time.perf_counter()-start, errors=errors,
            pending=[dict(variant=v,sequence=s) for v,s in pending]))
    status('running', jobs);report()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(one,v,s,manifest):(v,s) for v,s in jobs}
        while pending:
            finished, _ = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
            for f in finished:
                v,s = pending.pop(f)
                try:
                    done.append(f.result())
                except Exception as exc:
                    errors.append(dict(variant=v, sequence=s, error=repr(exc)))
                    print('FAILED',v,s,repr(exc),flush=True)
            status('running', pending.values())
            if finished:
                report()
    status('failed' if errors else 'complete', []);report()
    if errors:
        raise RuntimeError(errors)

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2)
    main(parser.parse_args().workers)
