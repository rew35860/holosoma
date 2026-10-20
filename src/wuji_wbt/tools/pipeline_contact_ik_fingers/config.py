"""Isolated organizer contact-first, adjustable-finger arm IK experiment."""
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
GUIDANCE_CODE = CODE.parent/'pipeline_v2_arm_guidance'
PILOT = ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_pilot_20261018'
OUT = ROOT/'holosoma/eval_out/ref_candidates/contact_first_ik_fingers_organizer_20261018'
SEQ = 'pick_up_and_carry_organizer_medium_while_walking-096'
PY = '/home/natcha/miniconda3/envs/omniretarget/bin/python'
FIXED = ROOT/'holosoma/eval_out/ref_candidates/contact_first_ik_organizer_20261018'
REMOVALS = {'baseline': {}, 'fixed_fingers': {}, 'contact_ik': {}}
LABELS = {'baseline': 'Saved V2', 'fixed_fingers': 'IK: fingers fixed', 'contact_ik': 'IK: arms + fingers'}
sys.path.append(str(PILOT_CODE))
from catalog import baseline_for


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj)
    return obj


def inputs(seq=SEQ):
    base=baseline_for(seq);job=read(base/'jobs'/f'{seq}.json')
    return dict(model=Path(job['model_xml']),object=job['object'],
        canonical=base/'clips'/seq/'part0'/f'{seq}_canonical.npz',
        baseline=PILOT/'clips'/seq/'final_v2.npz',
        human=PILOT/'human'/f'{seq}.npz',
        metrics=PILOT/'clips'/seq/'metrics.json',
        mask=base/'evaluation_v1/per_frame'/f'{seq}.npz')
