"""Step 3: regional surface contact with signed material penetration recovery."""
from pathlib import Path
import importlib.util
import sys

CODE = Path(__file__).resolve().parent
BASE_CODE = CODE.parent / 'pipeline_v2_contact_geometry'
SURFACE_CODE = CODE.parent / 'pipeline_v2_surface_contact'
spec = importlib.util.spec_from_file_location('signed_contact_base_config', BASE_CODE / 'config.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
for name in ('WORK', 'ROOT', 'RET', 'PILOT_CODE', 'GUIDANCE', 'PILOT', 'PY',
             'COHORT', 'EXTRA', 'read', 'write', 'sha', 'module', 'inputs', 'baseline_for'):
    globals()[name] = getattr(base, name)
BASE_OUT = ROOT / 'holosoma/eval_out/ref_candidates/pipeline_v2_surface_contact_20261019'
OUT = ROOT / 'holosoma/eval_out/ref_candidates/pipeline_v2_signed_contact_20261019'
REMOVALS = {'surface': (True, True), 'signed': (True, True)}
LABELS = {'surface': 'Surface-only baseline', 'signed': 'Surface + signed recovery'}
SIGNED_WEIGHT = 256.
sys.path.append(str(BASE_CODE))


def destination(seq, variant):
    return OUT / 'variants' / variant / 'clips' / seq
