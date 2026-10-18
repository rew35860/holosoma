"""Verify saved pilot outputs and tracker resampling without rerunning retargeting."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import mujoco

WORK = Path(__file__).resolve().parents[4]
OUT = WORK.parent/'holosoma/eval_out/ref_candidates/pipeline_v2_pilot_20261018'
BATCH = WORK.parent/'holosoma/eval_out/ref_candidates/batch_20261011_current_120'
spec = importlib.util.spec_from_file_location('converter', WORK/'src/wuji_wbt/motion/convert_wuji.py')
C = importlib.util.module_from_spec(spec);spec.loader.exec_module(C)

def read(p): return json.loads(p.read_text())

checks = []
for seq in read(OUT/'cohort.json')['sequences']:
    d = OUT/'clips'/seq
    run, metrics, done = [read(d/n) for n in ('run.json','metrics.json','completed.json')]
    model = mujoco.MjModel.from_xml_path(run['model'])
    q = np.load(d/'final_v2.npz')['qpos']
    init = np.load(d/'finger_initialization.npz')['qpos']
    assert q.shape == init.shape == (metrics['frames'], model.nq)
    assert np.isfinite(q).all()
    output = np.load(d/'v2_wuji_50hz.npz')
    resampled = C.resample_qpos(q, model, 30, 50)
    assert np.array_equal(output['joint_pos'], resampled[:, :-7])
    assert np.array_equal(output['object_pos_w'], resampled[:, -7:-4])
    assert np.array_equal(output['object_quat_w'], resampled[:, -4:])
    for name in output.files:
        a = output[name]
        if np.issubdtype(a.dtype, np.number): assert np.isfinite(a).all(), name
    assert output['fps'].item() == 50
    assert run['frozen_coordinates_exact'] and metrics['unchanged_body_exact']
    assert metrics['unchanged_source_contact_mask'] and run['final_uses_refined_fingers']
    assert metrics['finger_change_from_quick_init_max_rad'] > 1e-4
    assert run['solver_failures'] == 0
    assert hashlib.sha256((d/'final_v2.npz').read_bytes()).hexdigest() == done['final_sha256']
    checks.append(dict(sequence=seq, frames_30hz=len(q), frames_50hz=len(resampled),
                       direct_refined_export=True, exact_resampling=True,
                       remaining_body_exact=True, source_contact_mask_exact=True,
                       fingers_refined=True, all_values_finite=True,
                       code_sha256=run['code_sha256'],
                       repeated_body_max_difference=done['repeated_body_max_difference']))
assert len({r['code_sha256'] for r in checks}) == 1
assert len({json.dumps(read(OUT/'clips'/r['sequence']/'run.json')['settings'],sort_keys=True) for r in checks}) == 1
result = dict(passed=True, clips=checks, same_solver_code_and_settings_all_clips=True)
(OUT/'output_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
