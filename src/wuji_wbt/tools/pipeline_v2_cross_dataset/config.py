"""Isolated, CPU-only cross-dataset V2 pilot. Never changes production assets."""
from pathlib import Path
import importlib.util
import os
import sys

CODE = Path(__file__).resolve().parent
BACKEND_CODE = CODE.parent/'pipeline_v2_collision_backend'
spec=importlib.util.spec_from_file_location('cross_dataset_base',BACKEND_CODE/'config.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
for name in dir(base):
    if not name.startswith('_') and name not in ('CODE','spec','base'):
        globals()[name]=getattr(base,name)
OUT=Path(os.environ.get('V2_CROSS_OUT',str(ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_cross_dataset_20261021'))).resolve()
PILOT=OUT
RUN_VARIANT=os.environ.get('CROSS_VARIANT','current')
VARIANTS=('current','tip_relaxed','surface_emphasis')
REMOVALS={v:(True,True) for v in (*VARIANTS,'geometry','both','surface','mujoco','priority')}
LABELS={'current':'Current V2 + new init','tip_relaxed':'V2: weaker tip tracking',
        'surface_emphasis':'Tuned V3'}
NATIVE=ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_humoto_native_floor_10_20261021'
BATCH=ROOT/'holosoma/eval_out/ref_candidates/batch_20261011_current_120'
HIPHI=ROOT/'retarget_eval/hiphi_hand_shape_eval/human_comparison_20'
_loader=module
def module(name,path):
    if Path(path)==PILOT_CODE/'evaluate_one.py':path=CODE/'evaluate_source.py'
    return _loader(name,path)
def rows():return read(OUT/'cohort.json')
def row(seq):return next(r for r in rows() if r['sequence']==seq)
def baseline_for(seq):return OUT
def destination(seq,variant=None):return OUT/'variants'/RUN_VARIANT/'clips'/seq
def inputs(seq):
    r=row(seq);obj=r['object'];clip=OUT/'clips'/seq
    return dict(object=obj,model=clip/'full_model.xml',
        body=clip/'part1'/f'{seq}_stageA.npz',
        body_model=clip/'models'/seq/'stageA'/f'g1_29dof_wuji_welded_w_{obj}.xml',
        canonical=clip/'part0'/f'{seq}_canonical.npz',
        human=OUT/'human'/f'{seq}.npz',initialization=clip/'finger_initialization.npz',
        saved_v2=destination(seq)/'final_v2.npz',object_mesh=OUT/'geometry'/obj/'visual.obj',
        masks=OUT/'source_masks'/f'{seq}.npz')
def marker(seq,stage):
    folder=destination(seq) if stage in ('solver','evaluate') else OUT/'clips'/seq
    return folder/f'{stage}_completed.json'
def run_stage(stage,fn,seq):
    import fcntl,time
    lock=marker(seq,stage).with_suffix('.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('w') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        prepared=OUT/'clips'/seq/'canonical_completed.json'
        if prepared.exists() and read(prepared).get('source')!=row(seq):
            raise RuntimeError('Manifest changed after preparation. Use a new V2_CROSS_OUT directory; do not reuse stale outputs.')
        if marker(seq,stage).exists():return
        start=time.perf_counter();fn(seq)
        result=read(marker(seq,stage)) if marker(seq,stage).exists() else {}
        result.update(stage=stage,process_seconds=time.perf_counter()-start)
        write(marker(seq,stage),result)
