"""Matched current-V2 collision backend comparison; frozen ten-clip inputs."""
from pathlib import Path
import importlib.util
import sys

CODE = Path(__file__).resolve().parent
SHARED_CODE = CODE.parent / 'pipeline_v2_motion_init_10'
spec = importlib.util.spec_from_file_location('backend_shared_config', SHARED_CODE/'config.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
for name in dir(base):
    if not name.startswith('_') and name not in ('CODE', 'spec', 'base'):
        globals()[name] = getattr(base, name)
SHARED = OUT
OUT = ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_collision_backend_10_20261020'
PILOT = SHARED
VARIANTS = ('coal', 'mujoco')
SELECTED_VARIANT = 'mujoco'
REMOVALS = {v: (True, True) for v in (*VARIANTS, 'surface', 'both', 'geometry', 'priority')}
LABELS = dict(coal='Current V2: Coal', mujoco='Current V2: MuJoCo')

def destination(seq, variant):
    return OUT/'variants'/variant/'clips'/seq
