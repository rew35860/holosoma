"""Resource-limited four-condition comparison; baseline also rerun exactly."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
from config import *

def prepare():
    assert not OUT.exists(),'Never overwrite a completed or running experiment'
    old=read(PREVIOUS/'protocol.json')
    for group in ('source_sha256','input_sha256','target_sha256'):
        for path,expected in old[group].items():assert sha(path)==expected,(group,path)
    OUT.mkdir(parents=True);shutil.copytree(PREVIOUS/'targets',OUT/'targets')
    sources={Path(p) for p in old['source_sha256']}|set(CODE.glob('*.py'))
    write(OUT/'protocol.json',dict(version='motion-continuation-1',date='2026-10-20',sequences=COHORT,variants=LABELS,
        baseline='Exact fresh rerun of previous geometry-side / recovery-off / x1 outputs; equality required.',
        options=OPTIONS,signed_recovery=False,geometry_side=True,thumb_index_multiplier=1,
        unchanged='All objectives/weights, signed Coal collision pairs, 1 mm QP tolerance/slack1e5, shared Clarabel numerics, source endpoints/patches/masks, fixed Omni body and fixed scaled object trajectory, endpoint joint bounds and total 0.2 rad/frame limit.',
        motion_init='Contact-gated palm pose predictor transports previous robot palm with object rigid motion, then bounded damped 7-DoF arm IK (8 iterations max/hand). Finger angles unchanged in predictor; standard joint arm/finger solve follows.',
        substeps='Adaptive continuation of prescribed body/object transforms. Nominal <=3 mm tip-relative displacement and <=1 deg relative orientation per interval. Halve interval if signed warm depth grows >1.5 mm beyond max(previous pair depth,1 mm). Check 25/50/75/100% interpolated transition; halve if depth grows >1 mm beyond that reference. Not continuous collision detection. Min fraction1/64, max32 accepted/64 attempted steps; unresolved minimum steps or cap fallbacks explicitly logged.',
        interpolation='Translations/hinges linear; freejoint rotations on SO(3). Source hand targets interpolate in object frame. Current endpoint contact mask, patches, directions and skin-gap targets used on intermediate steps. Exact original targets restored at endpoint. Source evaluator never sees intermediate targets.',
        rate_budget='Each accepted interval fraction h receives <=0.2*h rad/joint relative to its preceding accepted pose; intervals sum to1. Original previous-frame temporal-prior target retained throughout. Up to15 SQP iterations per intermediate solve: added compute is reported, not concealed as an equal-iteration comparison.',
        evaluation='Existing historical, contact-geometry and geometry-side definitions unchanged. Full-clip results plus explicitly supplementary organizer frames105..120 and rack73..90 event windows, angular jerk/acceleration and guard failure counts.',
        resources=dict(workers=2,cpu_quota_percent=200,nice=19,memory_high_gib=4,memory_max_gib=5,memory_swap_max=0,gpu=False),
        source_sha256={str(p):sha(p) for p in sorted(sources)},input_sha256=old['input_sha256'],
        target_sha256={str(p):sha(p) for p in (OUT/'targets').glob('*.npz')}))
    for source in sources:
        dest=OUT/'source_snapshot'/source.relative_to(WORK);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)

def command(script,args,log):
    start=time.perf_counter()
    with Path(log).open('w') as f:
        subprocess.run([PY,str(CODE/script),*args],cwd=WORK,stdout=f,stderr=subprocess.STDOUT,check=True)
    return time.perf_counter()-start

def one(seq,variant):
    dest=destination(seq,variant);dest.mkdir(parents=True,exist_ok=True)
    solve=command('solver.py',[seq,variant],dest/'solver.log')
    evaluation=command('evaluate.py',[seq,variant],dest/'evaluation.log')
    result=dict(sequence=seq,variant=variant,solve_process_seconds=solve,evaluation_process_seconds=evaluation)
    write(dest/'completed.json',dict(state='complete',**result));return result

def validate():
    from solver import load
    protocol=read(OUT/'protocol.json');records=[]
    for group in ('source_sha256','input_sha256','target_sha256'):
        for path,expected in protocol[group].items():assert sha(path)==expected,(group,path)
    for seq in COHORT:
        definitions=None
        for variant in VARIANTS:
            dest=destination(seq,variant);Q=np.load(dest/'final_v2.npz')['qpos']
            _,_,_,ref,body,_=load(seq,variant);frozen=np.setdiff1d(np.arange(ref.m.nq),ref.qadr)
            assert np.array_equal(Q[:,frozen],body[:,frozen]) and np.isfinite(Q).all()
            assert np.all(Q[:,ref.qadr]>=ref.lo-1e-7) and np.all(Q[:,ref.qadr]<=ref.hi+1e-7)
            assert np.abs(np.diff(Q[:,ref.qadr],axis=0)).max()<=.2+1e-6
            metric=read(dest/'metrics.json');assert metric['unchanged_source_contact_mask'] and metric['unchanged_body_exact']
            dd=read(dest/'diagnostics.json')['definitions']
            if definitions is None:definitions=dd
            assert dd==definitions
            run=read(dest/'run.json');assert not run['signed_recovery']['enabled'] and run['thumb_index_contact_multiplier']==1
            records.append(dict(sequence=seq,variant=variant,fixed_body_object=True,source_contact_unchanged=True,limits_rates_passed=True))
    write(OUT/'validation.json',dict(passed=True,records=records,evaluation_definitions_unchanged=True))

def main():
    if not OUT.exists():prepare()
    resume='--resume' in sys.argv
    workers=int(sys.argv[sys.argv.index('--workers')+1]) if '--workers' in sys.argv else 2
    assert resume or not (OUT/'status.json').exists(),'Use --resume only after stopping this experiment service'
    elapsed_before=read(OUT/'status.json').get('elapsed_seconds',0.) if resume and (OUT/'status.json').exists() else 0.
    start=time.perf_counter()
    done=[read(destination(seq,v)/'completed.json') for v in VARIANTS for seq in COHORT if (destination(seq,v)/'completed.json').exists()]
    write(OUT/'status.json',dict(state='preflight',completed=len(done),total=8,workers=workers,elapsed_seconds=elapsed_before))
    if not (OUT/'preflight.json').exists():command('preflight.py',[],OUT/'preflight.log')
    errors=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(one,seq,v):(seq,v) for v in VARIANTS for seq in COHORT if not (destination(seq,v)/'completed.json').exists()}
        while pending:
            finished,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
            for future in finished:
                seq,v=pending.pop(future)
                try:done.append(future.result())
                except Exception as exc:errors.append(dict(sequence=seq,variant=v,error=repr(exc)))
            write(OUT/'status.json',dict(state='running',completed=len(done),total=8,workers=workers,pending=list(pending.values()),errors=errors,elapsed_seconds=elapsed_before+time.perf_counter()-start))
    if errors:raise RuntimeError(errors)
    validate();module('continuation_report',CODE/'report.py').main()
    module('continuation_sensitivity',BASE_CODE/'sensitivity.py').main()
    write(OUT/'status.json',dict(state='rendering',completed=8,total=8,timings=done,elapsed_seconds=elapsed_before+time.perf_counter()-start))
    with ThreadPoolExecutor(max_workers=2) as pool:
        renders=list(pool.map(lambda seq:command('render.py',[seq],OUT/f'{seq}_render.log'),COHORT))
    module('continuation_final_report',CODE/'report.py').main()
    write(OUT/'status.json',dict(state='complete',completed=8,total=8,timings=done,render_seconds=renders,elapsed_seconds=elapsed_before+time.perf_counter()-start))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():write(OUT/'status.json',dict(state='failed',error=repr(exc)))
        raise
