"""Matched geometry-side/recovery-off initialization and continuation pilot."""
from pathlib import Path
import importlib.util
import sys
CODE = Path(__file__).resolve().parent
GEOMETRY_CODE = CODE.parent/'pipeline_v2_geometry_side_no_recovery'
spec = importlib.util.spec_from_file_location('continuation_base_config', GEOMETRY_CODE/'config.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
for name in ('WORK','ROOT','RET','PILOT_CODE','GUIDANCE','PILOT','PY','BASE_CODE','SIGNED_CODE',
             'SURFACE_CODE','NUMERICS_CODE','COHORT','EXTRA','SIGNED_WEIGHT','read','write','sha','module','inputs','baseline_for'):
    globals()[name] = getattr(base,name)
PREVIOUS = base.OUT
OUT = ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_motion_continuation_20261020'
VARIANTS = ('baseline','motion_init','substeps','combined')
REMOVALS = {v:(True,True) for v in (*VARIANTS,'geometry','priority','surface','both')}
LABELS = dict(baseline='Geometry side; recovery OFF', motion_init='+ Motion-compensated init',
              substeps='+ Collision-aware substeps', combined='+ Init and substeps')
OPTIONS = dict(predictor_iterations=8, predictor_damping=1e-5, predictor_rotation_length_m=.1,
               predictor_max_joint_step_rad=.1, continuation_tip_step_m=.003,
               continuation_rotation_step_rad=.017453292519943295, warm_depth_increase_m=.0015,
               accepted_depth_increase_m=.001, minimum_fraction=1/64, max_accepted_substeps=32,
               max_transition_attempts=64, swept_check_fractions=[.25,.5,.75,1.])

def destination(seq,variant):
    return OUT/'variants'/variant/'clips'/seq
