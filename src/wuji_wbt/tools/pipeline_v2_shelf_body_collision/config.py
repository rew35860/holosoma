"""Original HUMOTO source, corrected floor, unchanged selected MuJoCo V2."""
from pathlib import Path
import importlib.util
import sys

CODE=Path(__file__).resolve().parent
SOURCE_CODE=CODE.parent/'pipeline_v2_humoto_smplx'
spec=importlib.util.spec_from_file_location('native_floor_parent',SOURCE_CODE/'config.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
for name in dir(base):
    if not name.startswith('_') and name not in ('CODE','SOURCE_CODE','spec','base'):
        globals()[name]=getattr(base,name)
PREVIOUS=ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_humoto_native_floor_10_20261021'
OUT=ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_shelf_body_collision_20261021'
COHORT=['shifting_shelf_with_left_leg-241']
PILOT=OUT
LABELS={'mujoco':'V2: body object collisions ON'}
WORKERS=1
_load_module=base._load_module

def module(name,path):
    if Path(path)==PILOT_CODE/'evaluate_one.py':path=CODE/'evaluate_source.py'
    return _load_module(name,path)

def baseline_for(seq):return OUT

def inputs(seq):
    old=native_inputs(seq);clip=OUT/'clips'/seq
    return dict(old,body=clip/'part1'/f'{seq}_stageA.npz',
        body_model=clip/'models'/seq/'stageA'/old['body_model'].name,
        canonical=clip/'part0'/f'{seq}_canonical.npz',human=OUT/'human'/f'{seq}.npz',
        initialization=clip/'finger_initialization.npz',masks=OUT/'source_masks'/f'{seq}.npz',
        saved_v2=destination(seq)/'final_v2.npz')

def destination(seq,variant='mujoco'):return OUT/'variants'/variant/'clips'/seq

def marker(seq,stage):
    return {'canonical':OUT/'clips'/seq/'canonical_completed.json',
        'body':OUT/'clips'/seq/'body_completed.json','prepare':OUT/'clips'/seq/'prepared.json',
        'solver':destination(seq)/'solve_completed.json','evaluate':destination(seq)/'completed.json',
        'render':OUT/'renders'/f'{seq}.json'}[stage]

def run_stage(stage,fn,seq):
    import fcntl,time
    lock=OUT/'locks'/f'{seq}_{stage}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        if marker(seq,stage).exists():return
        started=time.time();begin=time.perf_counter();fn(seq)
        elapsed=time.perf_counter()-begin
        write(OUT/'stage_execution'/seq/f'{stage}.json',dict(stage=stage,seconds=elapsed,
            started_unix=started,finished_unix=time.time()))
        if stage in ('solver','evaluate'):write(marker(seq,stage),dict(sequence=seq,stage=stage,process_seconds=elapsed))
        assert marker(seq,stage).exists(),marker(seq,stage)
