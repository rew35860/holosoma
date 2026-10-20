"""Two-clip, four-condition contact geometry pilot; no pad-facing objective."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

CODE = Path(__file__).resolve().parent
WORK = CODE.parents[3]
ROOT = WORK.parent
RET = WORK/'src/holosoma_retargeting/holosoma_retargeting'
PILOT_CODE = CODE.parent/'pipeline_v2_pilot'
GUIDANCE = CODE.parent/'pipeline_v2_arm_guidance'
PILOT = ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_pilot_20261018'
OUT = ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_contact_geometry_20261018'
PY = '/home/natcha/miniconda3/envs/omniretarget/bin/python'
sys.path.append(str(PILOT_CODE))
from catalog import baseline_for

COHORT = ['pick_up_and_carry_organizer_medium_while_walking-096',
          'dragging_turning__clothes_rack_walking_on_ground-802']
REMOVALS = {'control': (False, False), 'direction': (True, False),
            'gap': (False, True), 'both': (True, True)}
LABELS = {'control': 'V2 control (Coal)', 'direction': 'V2 + direction',
          'gap': 'V2 + surface gap', 'both': 'V2 + direction + gap'}
EXTRA = dict(direction_weight=2500., gap_weight=256., patch_radius_m=.02,
             patch_normal_cosine=.5, human_tip_region_radius_m=.02,
             contact_threshold_m=.03, source_zero_gap_m=.0001)

def read(path):
    return json.loads(Path(path).read_text())

def write(path, obj):
    path=Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n');temp.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def module(name, path):
    spec=importlib.util.spec_from_file_location(name, path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj)
    return obj

def inputs(seq):
    base=baseline_for(seq);job=read(base/'jobs'/f'{seq}.json');clip=base/'clips'/seq
    obj=job['object']
    return dict(object=obj, model=Path(job['model_xml']),
        body=clip/'part1'/f'{seq}_stageA.npz',
        body_model=clip/'models'/seq/'stageA'/f'g1_29dof_wuji_welded_w_{obj}.xml',
        canonical=clip/'part0'/f'{seq}_canonical.npz',
        human=PILOT/'human'/f'{seq}.npz',
        initialization=PILOT/'clips'/seq/'finger_initialization.npz',
        saved_v2=PILOT/'clips'/seq/'final_v2.npz',
        object_mesh=RET/'models'/obj/f'{obj}.obj',
        masks=base/'evaluation_v1/per_frame'/f'{seq}.npz')

def destination(seq, variant):
    return OUT/'variants'/variant/'clips'/seq
