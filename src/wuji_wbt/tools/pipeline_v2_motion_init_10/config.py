"""Ten-clip expansion of the unchanged geometry-side/motion-init comparison."""
from pathlib import Path
import importlib.util
import sys

CODE = Path(__file__).resolve().parent
IMPLEMENTATION_CODE = CODE.parent / "pipeline_v2_motion_continuation"
spec = importlib.util.spec_from_file_location("motion10_base_config", IMPLEMENTATION_CODE/"config.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
for name in ("WORK", "ROOT", "RET", "PILOT_CODE", "GUIDANCE", "PY", "BASE_CODE",
             "SIGNED_CODE", "SURFACE_CODE", "NUMERICS_CODE", "GEOMETRY_CODE",
             "EXTRA", "SIGNED_WEIGHT", "OPTIONS", "read", "write", "sha", "module",
             "baseline_for"):
    globals()[name] = getattr(base, name)
ARCHIVED_PILOT = base.PILOT
PREVIOUS = base.OUT
OUT = ROOT/"holosoma/eval_out/ref_candidates/pipeline_v2_motion_init_10_20261020"
PILOT = OUT  # Input skin/settings for this cohort; historical pilot stays untouched.
ORIGINAL_SIX = read(ARCHIVED_PILOT/"cohort.json")["sequences"]
ADDITIONAL_FOUR = [
    "carrying_working_chair_with_right_hand-839",
    "lifting_and_checking_step_stool-463",
    "drag_utility_cart_on_ground_while_walking_back_and_forth-677",
    "shifting_shelf_with_left_leg-241",
]
COHORT = ORIGINAL_SIX + ADDITIONAL_FOUR
VARIANTS = ("baseline", "motion_init")
REMOVALS = {v: (True, True) for v in (*VARIANTS, "both", "surface", "geometry", "priority")}
LABELS = dict(baseline="Geometry side; recovery OFF",
              motion_init="+ Motion-compensated init")
SETTINGS = read(ARCHIVED_PILOT/"clips"/ORIGINAL_SIX[0]/"run.json")["settings"]


def inputs(seq):
    source = baseline_for(seq)
    job = read(source/"jobs"/f"{seq}.json")
    clip, obj = source/"clips"/seq, job["object"]
    return dict(object=obj, model=Path(job["model_xml"]),
        body=clip/"part1"/f"{seq}_stageA.npz",
        body_model=clip/"models"/seq/"stageA"/f"g1_29dof_wuji_welded_w_{obj}.xml",
        canonical=clip/"part0"/f"{seq}_canonical.npz",
        human=OUT/"human"/f"{seq}.npz",
        initialization=OUT/"clips"/seq/"finger_initialization.npz",
        saved_v2=ARCHIVED_PILOT/"clips"/seq/"final_v2.npz",
        object_mesh=RET/"models"/obj/f"{obj}.obj",
        masks=source/"evaluation_v1/per_frame"/f"{seq}.npz")


def destination(seq, variant):
    return OUT/"variants"/variant/"clips"/seq
