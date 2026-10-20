"""Detached two-CPU six-solve experiment, evaluations and synchronized renders."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
from config import *


def prepare():
    assert not OUT.exists(),'Never overwrite a completed or running experiment'
    old=read(BASELINE/'protocol.json')
    for group in ('source_sha256','input_sha256','target_sha256'):
        for path,expected in old[group].items():assert sha(path)==expected,(group,path)
    OUT.mkdir(parents=True);shutil.copytree(BASELINE/'targets',OUT/'targets')
    sources={Path(p) for p in old['source_sha256']}|set(CODE.glob('*.py'))
    write(OUT/'protocol.json',dict(version='geometry-side-without-recovery-1',experiment_date='2026-10-20',
        branch='feature/wuji-pipeline-v2',sequences=COHORT,variants={v:dict(label=LABELS[v],
            side='reference point, zero plane allowance' if v=='baseline' else 'min distal collision vertex, 1 mm plane allowance',
            side_weight=2500.,gap_weight=256.,primary_multiplier=4 if v=='priority' else 1,
            primary_scope='Thumb/index side and gap only; both hands, active source contacts',
            signed_recovery=False,hard_side_constraint=False) for v in VARIANTS},
        objective_change='Geometry replaces only the reference-point side residual. Priority additionally scales the side and gap residuals by 2 for both hands thumb/index, i.e. squared cost ×4. There is no extra signed-recovery residual in any condition.',
        unchanged='Signed Coal collision backend and registered pairs, 1-mm collision tolerance and slack weight 1e5, surface patches/gap targets/30-mm source masks, human tracking and all other V2 weights, quick-finger initialization, original Omni body/object, joint/rate/trust limits, numerical solver and nonlinear line search.',
        inherited_geometry_change='As in the earlier geometry-side variant, replacing the point test also changes the plane allowance from zero to 1 mm. This is the requested existing variant, not an isolation of geometry representation alone.',
        numerical_policy=old['numerical_change'],fresh_solves=6,
        evaluation='Frozen historical evaluator and contact-geometry evaluator unchanged. Supplemental min-vertex side/contact and contact-tip penetration are explicitly named; no masks or evaluation thresholds replaced.',
        sensitivity_penetration_gates_mm=[.01,.1,.5,1.,1.5,2.,5.],
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
    protocol=read(OUT/'protocol.json')
    for group in ('source_sha256','input_sha256','target_sha256'):
        for path,expected in protocol[group].items():assert sha(path)==expected,(group,path)
    records=[]
    for seq in COHORT:
        ref_settings=ref_extra=ref_definitions=None
        for variant in VARIANTS:
            dest=destination(seq,variant);r=read(dest/'run.json');d=read(dest/'diagnostics.json')
            if ref_settings is None:ref_settings,ref_extra,ref_definitions=r['settings'],r['extra'],d['definitions']
            assert r['settings']==ref_settings and r['extra']==ref_extra and d['definitions']==ref_definitions
            assert not r['signed_recovery']['enabled'] and not r['hard_side_constraint']
            Q=np.load(dest/'final_v2.npz')['qpos'];_,_,_,ref,body,_=load(seq,variant)
            fixed=np.setdiff1d(np.arange(ref.m.nq),ref.qadr)
            assert Q.shape==body.shape and np.isfinite(Q).all() and np.array_equal(Q[:,fixed],body[:,fixed])
            assert np.all(Q[:,ref.qadr]>=ref.lo-1e-7) and np.all(Q[:,ref.qadr]<=ref.hi+1e-7)
            assert np.abs(np.diff(Q[:,ref.qadr],axis=0)).max()<=ref.cfg.joint_rate_limit+1e-6
            metrics=read(dest/'metrics.json');assert metrics['unchanged_source_contact_mask'] and metrics['unchanged_body_exact']
            steps=[s for frame in read(dest/'iterations.json') for s in frame['steps']]
            assert all(s['numerical_objective_scale']==1. and s['maximum_cone_constraint_violation']<=1e-6 for s in steps)
            records.append(dict(sequence=seq,variant=variant,frames=len(Q),qp_steps=len(steps),
                certified_fallbacks=sum(s['accepted_feasible_fallback'] for s in steps),
                max_qp_constraint_violation=max(s['maximum_cone_constraint_violation'] for s in steps),
                fixed_body_object_exact=True,source_mask_unchanged=True,limits_and_rates_passed=True,
                signed_recovery_off=True,hard_side_constraint_off=True))
    write(OUT/'validation.json',dict(passed=True,sequences=records,
        preflight=read(OUT/'preflight.json'),evaluation_definitions_unchanged=True))


def validate_renders():
    import imageio_ffmpeg
    from PIL import Image
    records=[]
    for seq in COHORT:
        T=read(destination(seq,'baseline')/'run.json')['frames']
        for suffix in ('','_hands'):
            path=OUT/'renders'/f'{seq}{suffix}.mp4'
            decoder=imageio_ffmpeg.read_frames(str(path),input_params=['-threads','1'],output_params=['-threads','1'])
            meta=next(decoder);count=sum(1 for _ in decoder)
            assert count==(T+1)//2 and meta['size']==(1920,540),(seq,count,meta)
            records.append(dict(path=str(path),frames=count,size=meta['size'],fps=meta['fps'],sha256=sha(path)))
        for suffix in ('','_hands','_posture'):
            with Image.open(OUT/'renders'/f'{seq}{suffix}.png') as im:im.verify()
    write(OUT/'render_validation.json',dict(passed=True,videos=records))


def main():
    prepare();start=time.perf_counter();write(OUT/'status.json',dict(state='preflight',completed=0,total=6))
    command('preflight.py',[],OUT/'preflight.log')
    done=[];errors=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(one,seq,v):(seq,v) for v in VARIANTS for seq in COHORT}
        while pending:
            finished,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
            for future in finished:
                seq,v=pending.pop(future)
                try:done.append(future.result())
                except Exception as exc:errors.append(dict(sequence=seq,variant=v,error=repr(exc)))
            write(OUT/'status.json',dict(state='running',completed=len(done),total=6,pending=list(pending.values()),
                errors=errors,elapsed_seconds=time.perf_counter()-start))
    if errors:raise RuntimeError(errors)
    validate()
    module('geometry_no_recovery_sensitivity',BASE_CODE/'sensitivity.py').main()
    module('geometry_no_recovery_report',CODE/'report.py').main()
    write(OUT/'status.json',dict(state='rendering',completed=6,total=6,validation_passed=True,
        timings=done,elapsed_seconds=time.perf_counter()-start))
    with ThreadPoolExecutor(max_workers=2) as pool:
        renders=list(pool.map(lambda seq:command('render.py',[seq],OUT/f'{seq}_render.log'),COHORT))
    validate_renders()
    module('geometry_no_recovery_report_final',CODE/'report.py').main()
    write(OUT/'status.json',dict(state='complete',completed=6,total=6,validation_passed=True,
        render_validation_passed=True,timings=done,render_seconds=renders,elapsed_seconds=time.perf_counter()-start))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            state=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
            write(OUT/'status.json',dict(state,state='failed',error=repr(exc)))
        raise
