"""Three matched contact variants; signed recovery is absent in every variant."""
from pathlib import Path
import importlib.util
import sys

CODE = Path(__file__).resolve().parent
NUMERICS_CODE = CODE.parent / 'pipeline_v2_solver_numerics'
spec = importlib.util.spec_from_file_location('geometry_no_recovery_base', NUMERICS_CODE / 'config.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
for name in ('WORK', 'ROOT', 'RET', 'PILOT_CODE', 'GUIDANCE', 'PILOT', 'PY',
             'BASE_CODE', 'SIGNED_CODE', 'SURFACE_CODE', 'COHORT', 'EXTRA', 'SIGNED_WEIGHT',
             'read', 'write', 'sha', 'module', 'inputs', 'baseline_for'):
    globals()[name] = getattr(base, name)
BASELINE = base.OUT
OUT = ROOT / 'holosoma/eval_out/ref_candidates/pipeline_v2_geometry_side_no_recovery_20261020'
VARIANTS = ('baseline', 'geometry', 'priority')
REMOVALS = {v:(True, True) for v in (*VARIANTS, 'surface', 'both')}
LABELS = {'baseline':'V2 surface gap; recovery OFF', 'geometry':'Geometry side; recovery OFF',
          'priority':'Geometry side + thumb/index x4'}


def destination(seq, variant):
    return OUT / 'variants' / variant / 'clips' / seq
