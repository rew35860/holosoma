"""Organizer contact-side diagnosis and matched thumb/index priority pilot."""
from pathlib import Path
import importlib.util
import sys

CODE = Path(__file__).resolve().parent
PARENT_CODE = CODE.parent / 'pipeline_v2_thumb_contact'
FOLLOWUP_CODE = CODE.parent / 'pipeline_v2_thumb_contact_followup'
NUMERICS_CODE = CODE.parent / 'pipeline_v2_solver_numerics'
spec = importlib.util.spec_from_file_location('thumb_constraint_config', FOLLOWUP_CODE / 'config.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
for name in ('WORK', 'ROOT', 'RET', 'PILOT_CODE', 'GUIDANCE', 'PILOT', 'PY',
             'BASE_CODE', 'SIGNED_CODE', 'SURFACE_CODE', 'EXTRA', 'SIGNED_WEIGHT',
             'read', 'write', 'sha', 'module', 'inputs', 'baseline_for'):
    globals()[name] = getattr(base, name)
BASELINE = base.OUT
OUT = ROOT / 'holosoma/eval_out/ref_candidates/pipeline_v2_thumb_side_constraint_20261019'
COHORT = [base.COHORT[0]]
REMOVALS = {v: (True, True) for v in ('baseline', 'combined', 'hard', 'signed')}
LABELS = {'baseline': 'Current signed contact', 'combined': 'Geometry side + thumb/index x4',
          'hard': 'Prevent wrong-side thumb' , 'signed': 'Diagnostic alias'}


def destination(seq, variant):
    return OUT / 'variants' / variant / 'clips' / seq
