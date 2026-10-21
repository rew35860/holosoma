"""Fresh SMPL-X inputs, frozen current MuJoCo-distance V2, ten matched clips."""
from pathlib import Path
import importlib.util
import sys

CODE = Path(__file__).resolve().parent
BACKEND_CODE = CODE.parent / 'pipeline_v2_collision_backend'
spec = importlib.util.spec_from_file_location('smplx_existing_config', BACKEND_CODE/'config.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
for name in dir(base):
    if not name.startswith('_') and name not in ('CODE', 'spec', 'base'):
        globals()[name] = getattr(base, name)
NATIVE_OUT = OUT
native_inputs = inputs
native_baseline_for = baseline_for
OUT = ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_humoto_smplx_10_20261020'
PILOT = OUT
FIT_ROOT = ROOT/'hiphi2smplx-cmp-humoto'
FIT_PY = '/home/natcha/miniconda3/envs/hiphi2smplx/bin/python'
MODEL = ROOT/'hiphi2smplx/assets/models/SMPLX_NEUTRAL.npz'
VARIANTS = ('mujoco',)
REMOVALS = {v: (True, True) for v in (*VARIANTS, 'surface', 'both', 'geometry', 'priority')}
LABELS = {'mujoco': 'Current V2: MuJoCo'}
sys.path.append(str(BASE_CODE))  # unchanged regional geometry helper
_load_module = module

def write(path, value):
    """Atomic metadata updates, including when workers meet at a cached clip."""
    import json
    import tempfile
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
        temporary=Path(stream.name)
    temporary.replace(path)

def module(name, path):
    # The historical wrapper assumes an old V1 result and its source labels.
    # Replace only that wrapper; preserve the underlying metric functions.
    if Path(path) == PILOT_CODE/'evaluate_one.py':
        path = CODE/'evaluate_source.py'
    return _load_module(name, path)

def baseline_for(seq):
    return OUT

def inputs(seq):
    old = native_inputs(seq)
    clip = OUT/'clips'/seq
    return dict(old, body=clip/'part1'/f'{seq}_stageA.npz',
        body_model=clip/'models'/seq/'stageA'/old['body_model'].name,
        canonical=clip/'part0'/f'{seq}_canonical.npz',
        human=OUT/'human'/f'{seq}.npz',
        initialization=clip/'finger_initialization.npz',
        masks=OUT/'source_masks'/f'{seq}.npz',
        saved_v2=destination(seq, 'mujoco')/'final_v2.npz')

def destination(seq, variant='mujoco'):
    return OUT/'variants'/variant/'clips'/seq

def run_stage(stage, fn, seq):
    """Coordinate independent background workers without duplicate computation."""
    import fcntl
    import time
    marker={'fit':OUT/'fits'/seq/'fit_report.json',
            'body':OUT/'clips'/seq/'body_completed.json',
            'prepare':OUT/'clips'/seq/'prepared.json',
            'solver':destination(seq)/'solve_completed.json',
            'evaluate':destination(seq)/'completed.json',
            'render':OUT/'renders'/f'{seq}.json'}[stage]
    lock=OUT/'locks'/f'{seq}_{stage}.lock';lock.parent.mkdir(exist_ok=True)
    with lock.open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        if marker.exists():return
        started=time.time();begin=time.perf_counter();fn(seq)
        write(OUT/'stage_execution'/seq/f'{stage}.json',dict(stage=stage,
            seconds=time.perf_counter()-begin,started_unix=started,finished_unix=time.time(),
            scope='Actual stage execution, excluding time waiting for another worker or Python startup'))
        if stage in ('solver','evaluate'):
            write(marker,dict(sequence=seq,stage=stage,process_seconds=time.perf_counter()-begin))
        assert marker.exists(),marker
