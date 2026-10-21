"""Matched collider comparison, native HUMOTO and unchanged current V2."""
from pathlib import Path
import importlib.util
import os
import sys

CODE = Path(__file__).resolve().parent
SOURCE_CODE = CODE.parent / 'pipeline_v2_shelf_body_collision'
spec = importlib.util.spec_from_file_location('proxy_motion_base', SOURCE_CODE / 'config.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
for name in dir(base):
    if not name.startswith('_') and name not in ('CODE', 'SOURCE_CODE', 'spec', 'base'):
        globals()[name] = getattr(base, name)
EXP = ROOT / 'holosoma/eval_out/ref_candidates/pipeline_v2_proxy_motion_20261021'
CHOICE = os.environ.get('PROXY_VARIANT', 'preset32')
assert CHOICE in ('preset32', 'components')
OUT = EXP / CHOICE
PILOT = OUT
PREVIOUS = ROOT / 'holosoma/eval_out/ref_candidates/pipeline_v2_humoto_native_floor_10_20261021'
BUDGET = ROOT / 'holosoma/eval_out/ref_candidates/pipeline_v2_proxy_budget_20261021'
AUDIT = ROOT / 'holosoma/eval_out/ref_candidates/pipeline_v2_thin_proxy_audit_20261021'
COHORT = ['drag_utility_cart_on_ground_while_walking_back_and_forth-677', 'lifting_and_checking_step_stool-463']
LABELS = {'preset32': 'UMR/GRAIL preset (32 pieces)', 'components': 'Component-based colliders'}
_load_module = base._load_module

def module(name, path):
    if Path(path) == PILOT_CODE / 'evaluate_one.py':
        path = CODE / 'evaluate_source.py'
    return _load_module(name, path)

def baseline_for(seq): return OUT

def inputs(seq):
    old = native_inputs(seq); clip = OUT / 'clips' / seq
    return dict(old, model=clip / 'full_model.xml',
        body=clip / 'part1' / f'{seq}_stageA.npz',
        body_model=clip / 'models' / seq / 'stageA' / old['body_model'].name,
        canonical=clip / 'part0' / f'{seq}_canonical.npz',
        human=OUT / 'human' / f'{seq}.npz', initialization=clip / 'finger_initialization.npz',
        masks=OUT / 'source_masks' / f'{seq}.npz', saved_v2=destination(seq) / 'final_v2.npz')

def destination(seq, variant='mujoco'): return OUT / 'variants' / variant / 'clips' / seq

def marker(seq, stage):
    return {'canonical': OUT/'clips'/seq/'canonical_completed.json',
        'body': OUT/'clips'/seq/'body_completed.json', 'prepare': OUT/'clips'/seq/'prepared.json',
        'solver': destination(seq)/'solve_completed.json', 'evaluate': destination(seq)/'completed.json'}[stage]

def run_stage(stage, fn, seq):
    import fcntl, time
    lock = OUT/'locks'/f'{seq}_{stage}.lock'; lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('w') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if marker(seq,stage).exists(): return
        start=time.perf_counter(); cpu=time.process_time(); stamp=time.time()
        fn(seq)
        result=dict(stage=stage, seconds=time.perf_counter()-start,
                    cpu_seconds=time.process_time()-cpu, started_unix=stamp, finished_unix=time.time())
        write(OUT/'stage_execution'/seq/f'{stage}.json', result)
        if stage in ('solver','evaluate'): write(marker(seq,stage), result)
        assert marker(seq,stage).exists()
