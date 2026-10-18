"""Fixed cohort and settings for the requested V2 objective ablation."""
from pathlib import Path
import hashlib
import json
import sys

CODE = Path(__file__).resolve().parent
WORK = CODE.parents[3]
ROOT = WORK.parent
PILOT_CODE = CODE.parent/'pipeline_v2_pilot'
sys.path.insert(0, str(PILOT_CODE))
from catalog import baseline_for

PILOT = ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_pilot_20261018'
OUT = ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_ablation_20261018'
RET = WORK/'src/holosoma_retargeting/holosoma_retargeting'
SOLVER = RET/'retarget_pipeline/pipeline_v2.py'
PY = '/home/natcha/miniconda3/envs/omniretarget/bin/python'

def read(p):
    return json.loads(Path(p).read_text())

def write(p, data):
    p = Path(p);p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n');tmp.replace(p)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

COHORT = read(PILOT/'cohort.json')['sequences']
BASE_SETTINGS = read(PILOT/'clips'/COHORT[0]/'run.json')['settings']
REMOVALS = {
    'full': {},
    'no_finger_prior': {'finger_prior_weight': 0.},
    'no_contact': {'contact_weight': 0.},
    'no_palm': {'palm_axis_weight': 0.},
    'no_fingertip': {'fingertip_weight': 0.},
    'no_arm_prior': {'arm_prior_contact': 0., 'arm_prior_no_contact': 0.},
    'core_only': {'fingertip_weight': 0., 'contact_weight': 0., 'palm_axis_weight': 0.,
                  'finger_prior_weight': 0., 'arm_prior_contact': 0., 'arm_prior_no_contact': 0.},
}
LABELS = {
    'full': 'Full V2', 'no_finger_prior': 'Without finger initialization prior',
    'no_contact': 'Without source-contact attraction', 'no_palm': 'Without palm alignment',
    'no_fingertip': 'Without world fingertip tracking', 'no_arm_prior': 'Without Omni arm prior (both weights)',
    'core_only': 'All five additions removed',
}

def inputs(seq):
    base = baseline_for(seq);job = read(base/'jobs'/f'{seq}.json')
    clip = base/'clips'/seq;obj = job['object']
    return dict(object=obj, canonical=clip/'part0'/f'{seq}_canonical.npz',
                body=clip/'part1'/f'{seq}_stageA.npz', model=Path(job['model_xml']),
                body_model=clip/'models'/seq/'stageA'/f'g1_29dof_wuji_welded_w_{obj}.xml')

def settings(variant):
    return {**BASE_SETTINGS, **REMOVALS[variant]}
