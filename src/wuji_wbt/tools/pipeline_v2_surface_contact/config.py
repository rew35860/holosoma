"""Step 2: remove legacy point attraction from the saved combined condition."""
from pathlib import Path
import importlib.util
import sys

CODE = Path(__file__).resolve().parent
BASE_CODE = CODE.parent / 'pipeline_v2_contact_geometry'
spec = importlib.util.spec_from_file_location('surface_contact_base_config', BASE_CODE / 'config.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
for name in ('WORK', 'ROOT', 'RET', 'PILOT_CODE', 'GUIDANCE', 'PILOT', 'PY',
             'COHORT', 'EXTRA', 'read', 'write', 'sha', 'module', 'inputs', 'baseline_for'):
    globals()[name] = getattr(base, name)
BASE_OUT = base.OUT
OUT = ROOT / 'holosoma/eval_out/ref_candidates/pipeline_v2_surface_contact_20261019'
REMOVALS = {'both': (True, True), 'surface': (True, True)}
LABELS = {'both': 'Combined: point + surface targets',
          'surface': 'Combined: surface target only'}
sys.path.append(str(BASE_CODE))


def destination(seq, variant):
    return OUT / 'variants' / variant / 'clips' / seq
