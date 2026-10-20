"""Isolated numerical regression; contact objectives and metrics are unchanged."""
from pathlib import Path
import importlib.util
import sys

CODE = Path(__file__).resolve().parent
SIGNED_CODE = CODE.parent / 'pipeline_v2_signed_contact'
spec = importlib.util.spec_from_file_location('numerics_base_config', SIGNED_CODE / 'config.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
for name in ('WORK', 'ROOT', 'RET', 'PILOT_CODE', 'GUIDANCE', 'PILOT', 'PY',
             'BASE_CODE', 'SURFACE_CODE', 'COHORT', 'EXTRA', 'SIGNED_WEIGHT',
             'REMOVALS', 'read', 'write', 'sha', 'module', 'inputs', 'baseline_for'):
    globals()[name] = getattr(base, name)
PREVIOUS = base.OUT
OUT = ROOT / 'holosoma/eval_out/ref_candidates/pipeline_v2_solver_numerics_20261019'
LABELS = {'signed': 'Explicit tolerances + certified fallback', 'surface': 'Diagnostic alias'}
ARCHIVED_QP = PREVIOUS / 'variants/signed/clips' / COHORT[0] / 'failed_subproblems/frame_0031_attempt_000.npz'


def destination(seq, variant):
    return OUT / 'variants' / variant / 'clips' / seq
