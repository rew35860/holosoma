"""Extend the paired pilot using the existing v1 references, as explicitly requested."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import hashlib
import json
import os
import shutil
import subprocess
import time
import numpy as np
from catalog import WORK, ROOT, OUT, ORIGINAL_BATCH, ADDITIONAL, baseline_for

RET=WORK/'src/holosoma_retargeting/holosoma_retargeting'
OLD=ROOT/'holosoma/src/holosoma_retargeting/holosoma_retargeting'
CODE=Path(__file__).resolve().parent
PY='/home/natcha/miniconda3/envs/omniretarget/bin/python'

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def read(p):return json.loads(p.read_text())
def run(args,log):
    start=time.perf_counter()
    with log.open('a') as f:
        f.write('COMMAND '+repr(args)+'\n');f.flush()
        subprocess.run(args,cwd=RET,stdout=f,stderr=subprocess.STDOUT,check=True)
    return time.perf_counter()-start

def one(seq):
    obj='table' if seq==ADDITIONAL[0] else 'organizer_medium'
    base=baseline_for(seq);clip=base/'clips'/seq;dest=OUT/'clips'/seq
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'completed.json').exists():return read(dest/'completed.json')
    for part,suffix in [('part0','canonical'),('part1','stageA'),('part2','stageB')]:
        f=f'{seq}_{suffix}.npz';target=clip/part/f;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(OLD/'demo_results/g1/retarget_pipeline'/part/f,target)
    models={
        'stageA':f'g1_29dof_wuji_welded_w_{obj}.xml',
        'stageB':f'hand_both_w_{obj}_coacd_ball.xml',
        'final':f'g1_29dof_wuji_w_{obj}_coacd.xml'}
    for stage,name in models.items():
        target=clip/'models'/seq/stage/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(OLD/'models/scaled'/stage/name,target)
    body=clip/'part1'/f'{seq}_stageA.npz';canonical=clip/'part0'/f'{seq}_canonical.npz'
    model=clip/'models'/seq/'final'/models['final'];bodymodel=clip/'models'/seq/'stageA'/models['stageA']
    inp=ROOT/'Data/InterAct/result/humoto_pt'
    bodyout=dest/'body_rerun'
    for source,target in [(canonical,bodyout/'part0'/canonical.name),(bodymodel,bodyout/'models'/seq/'stageA'/bodymodel.name)]:
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    def pipeline(stage,out):return [PY,str(RET/'retarget_pipeline/pipeline.py'),'--task-name',seq,
        '--object-name',obj,'--stage',stage,'--data-path',str(inp),'--save-dir',str(out)]
    times=read(dest/'extension_times.json') if (dest/'extension_times.json').exists() else {}
    if 'body' not in times or 'v2' not in times:
        with ThreadPoolExecutor(max_workers=2) as pool:
            a=pool.submit(run,pipeline('part1',bodyout),dest/'body.log')
            b=pool.submit(run,[PY,str(RET/'retarget_pipeline/pipeline_v2.py'),'--canonical',str(canonical),
                 '--body',str(body),'--body-model',str(bodymodel),'--model',str(model),'--object',obj,
                 '--output',str(dest)],dest/'refinement.log')
            times.update(body=a.result(),v2=b.result())
        write(dest/'extension_times.json',times)
    final=clip/'part3'/f'{seq}_final_separate.npz'
    final.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(OLD/'demo_results/g1/retarget_pipeline/part3'/final.name,final)
    stages={k:dict(wall_seconds=None,reused=True) for k in ('part1','part2','part3','convert50hz')}
    write(base/'jobs'/f'{seq}.json',dict(sequence=seq,object=obj,status='success',final_npz=str(final),
          model_xml=str(model),stages=stages,body_pen_tolerance=.001,
          baseline_note='Existing archived v1 output reused unchanged per user request; no v1 retargeting rerun. Organizer historical arm stage omitted object collision lookup. Historical per-stage runtimes unavailable; v1 timing undefined.'))
    evaluation=base/'evaluation_v1';evaluation.mkdir(parents=True,exist_ok=True)
    if not (evaluation/'protocol.json').exists():shutil.copy2(ORIGINAL_BATCH/'evaluation_v1/protocol.json',evaluation/'protocol.json')
    # Isolated process: unchanged metric implementation, only input/output locations differ.
    evaluate_script=CODE/'evaluate_added_baseline.py'
    t1=run([PY,str(evaluate_script),seq],dest/'v1_evaluation.log')
    t2=run([PY,str(CODE/'evaluate_one.py'),seq],dest/'evaluation.log')
    export=run([PY,str(WORK/'src/wuji_wbt/motion/convert_wuji.py'),str(dest/'final_v2.npz'),
                str(model),str(dest/'v2_wuji_50hz.npz'),'30','50'],dest/'export.log')
    before=np.load(body)['qpos'];after=np.load(bodyout/'part1'/body.name)['qpos'];assert before.shape==after.shape
    result=dict(sequence=seq,status='complete',measured_body_seconds=times['body'],
        measured_refinement_process_seconds=times['v2'],measured_export_seconds=export,
        total_sequential_retarget_export_seconds=times['body']+times['v2']+export,
        evaluation_seconds=t1+t2,repeated_body_max_difference=float(np.abs(before-after).max()),
        paired_refinement_uses_exact_archived_body=True,
        baseline_stage_seconds={k:v['wall_seconds'] for k,v in stages.items()},
        baseline_floating_hands_reused=True,baseline_full_runtime_available=False,
        final_sha256=hashlib.sha256((dest/'final_v2.npz').read_bytes()).hexdigest())
    write(dest/'completed.json',result);print('COMPLETE',seq,flush=True);return result

if __name__=='__main__':
    start=time.perf_counter()
    write(OUT/'extension_status.json',dict(state='running',sequences=list(ADDITIONAL)))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(one,ADDITIONAL))
        write(OUT/'extension_status.json',dict(state='complete',completed=2,wall_seconds=time.perf_counter()-start))
    except Exception as exc:
        write(OUT/'extension_status.json',dict(state='failed',error=repr(exc)));raise
