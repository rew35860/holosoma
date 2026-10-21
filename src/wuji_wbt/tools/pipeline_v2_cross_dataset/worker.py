"""Resumable capped background pilot; failures are recorded, never discarded."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import subprocess,time,traceback,concurrent.futures
from config import *

def stage(seq,name,variant='current'):
    folder=OUT/'logs'/seq;folder.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CROSS_VARIANT=variant,PROXY_BODY_RECOVERY='1')
    stamp=time.perf_counter()
    with (folder/f'{variant}_{name}.log').open('a') as log:
        result=subprocess.run([PY,str(CODE/f'{name}.py'),seq],env=env,stdout=log,stderr=subprocess.STDOUT,cwd=WORK)
    entry=dict(sequence=seq,variant=variant,stage=name,returncode=result.returncode,seconds=time.perf_counter()-stamp)
    write(OUT/'attempts'/f'{seq}_{variant}_{name}.json',entry)
    return result.returncode==0

def process(r,variant='current'):
    seq=r['sequence']
    if variant!='current' and not (OUT/'clips'/seq/'prepare_completed.json').exists():
        return dict(sequence=seq,variant=variant,status='blocked_by_upstream_failure',stage='prepare')
    names=['prepare_inputs','body','prepare','solver','evaluate'] if variant=='current' else ['solver','evaluate']
    for name in names:
        if not stage(seq,name,variant):return dict(sequence=seq,variant=variant,status='failed',stage=name)
    return dict(sequence=seq,variant=variant,status='complete')

def phase(label,jobs):
    outcomes=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(process,r,v) for r,v in jobs]
        for future in concurrent.futures.as_completed(futures):
            outcomes.append(future.result());write(OUT/'status.json',dict(state='running',phase=label,finished=len(outcomes),total=len(jobs),outcomes=outcomes))
    write(OUT/f'{label}_outcomes.json',outcomes)

def choose():
    import numpy as np
    tune=[r for r in rows() if r['split']=='tune'];stats={}
    for v in VARIANTS:
        values=[]
        for r in tune:
            folder=OUT/'variants'/v/'clips'/r['sequence']
            if (folder/'evaluate_completed.json').exists():
                d=read(folder/'diagnostics.json');g=read(folder/'geometry_diagnostics.json')
                values.append([d['physical_patch_gap_mean_mm'],d['object_max_penetration_mm'],d['arm_jerk_rms_rad_s3'],
                    d['self_max_penetration_mm'],d['physical_touch_vs_original_source']['f1_pct'],
                    g['geometry_intended_contact_pct'],d['arm_contact_near_limit5deg_pct']])
        keys=('mean_gap_mm','mean_peak_pen_mm','mean_arm_jerk','mean_self_peak_mm','mean_physical_f1_pct','mean_intended_contact_pct','near_arm_limit_pct')
        stats[v]=dict(successes=len(values),**{k:None for k in keys})
        if values:
            stats[v].update(zip(keys,np.mean(values,axis=0).tolist()))
    current=stats['current'];eligible=[]
    # Require all four tuning clips for a defensible shared choice.
    if current['successes']==len(tune):
        for v in VARIANTS[1:]:
            x=stats[v]
            if (x['successes']==len(tune) and x['mean_peak_pen_mm']<=current['mean_peak_pen_mm']+1
                and x['mean_self_peak_mm']<=current['mean_self_peak_mm']+1
                and x['mean_physical_f1_pct']>=current['mean_physical_f1_pct']-2
                and x['mean_intended_contact_pct']>=current['mean_intended_contact_pct']-2
                and x['near_arm_limit_pct']<=current['near_arm_limit_pct']+5
                and x['mean_arm_jerk']<=current['mean_arm_jerk']*1.25 and x['mean_gap_mm']<current['mean_gap_mm']):
                eligible.append(v)
    selected=min(eligible,key=lambda v:stats[v]['mean_gap_mm']) if eligible else 'current'
    result=dict(selected=selected,tuning=stats,locked_before_validation_candidate=True,
        reason='Lowest tuning patch gap satisfying predeclared solve/penetration/jerk checks' if eligible else 'No qualifying shared weight change; retain current')
    write(OUT/'selection.json',result);return selected

def main():
    start=time.time()
    write(OUT/'status.json',dict(state='running',phase='current',finished=0,total=len(rows())))
    phase('current',[(r,'current') for r in rows()])
    phase('tuning',[(r,v) for v in VARIANTS[1:] for r in rows() if r['split']=='tune'])
    selected=choose()
    if selected!='current':phase('validation',[(r,selected) for r in rows() if r['split']=='validation'])
    subprocess.run([PY,str(CODE/'report.py')],check=True)
    write(OUT/'status.json',dict(state='metrics_complete',selected=selected,seconds=time.time()-start,
        note='All attempts recorded; see report for failures. Renders are a subsequent stage.'))

if __name__=='__main__':
    try:main()
    except Exception:
        write(OUT/'worker_failure.json',dict(traceback=traceback.format_exc()));raise
