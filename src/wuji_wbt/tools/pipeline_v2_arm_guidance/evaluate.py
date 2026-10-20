"""Route each ablation through the unchanged pilot evaluator."""
import sys
import numpy as np
import mujoco
from config import OUT, PILOT, PILOT_CODE, read, write, sha, inputs, settings

sys.path.insert(0, str(PILOT_CODE))
import evaluate_one as pilot_eval

def main(variant, seq):
    dest = OUT/'variants'/variant/'clips'/seq
    pilot_eval.OUT = OUT/'variants'/variant
    pilot_eval.main(seq)
    q = np.load(dest/'final_v2.npz')['qpos']
    init = np.load(dest/'finger_initialization.npz')['qpos']
    old_init = np.load(PILOT/'clips'/seq/'finger_initialization.npz')['qpos']
    assert np.array_equal(init, old_init), 'Quick initialization changed'
    run = read(dest/'run.json');assert run['settings'] == settings(variant)
    info = inputs(seq);m = mujoco.MjModel.from_xml_path(str(info['model']))
    names = pilot_eval.E.joints(m)
    selected = [a for name, (a, _) in names.items() if 'finger' in name or
                any(name == f'{side}_{joint}_joint' for side in ('left', 'right') for joint in
                    ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw'))]
    assert len(selected) == 54 and np.isfinite(q).all()
    max_rate = float(np.max(np.abs(np.diff(q[:, selected], axis=0))))
    assert max_rate <= run['settings']['joint_rate_limit']+1e-6
    for j in range(m.njnt):
        a = int(m.jnt_qposadr[j])
        if a in selected:
            assert q[:, a].min() >= m.jnt_range[j, 0]-1e-6
            assert q[:, a].max() <= m.jnt_range[j, 1]+1e-6
    old = np.load(PILOT/'clips'/seq/'final_v2.npz')['qpos']
    difference = float(np.max(np.abs(q-old)))
    if variant == 'full':
        assert difference < 1e-7, 'Full V2 no longer reproduces the archived trajectory'
    metric = read(dest/'metrics.json')
    write(dest/'checks.json', dict(passed=True, initialization_exact=True,
        remaining_body_exact=metric['unchanged_body_exact'],
        source_contact_mask_exact=metric['unchanged_source_contact_mask'],
        settings_exact=True, finite=True, joint_limits_passed=True,
        maximum_joint_change_rad_per_frame=max_rate,
        full_v2_trajectory_max_difference=difference if variant == 'full' else None,
        solver_code_sha256=run['code_sha256'], evaluator_sha256=metric['evaluator_sha256'],
        adapter_sha256=sha(PILOT_CODE/'evaluate_one.py')))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
    import diagnostics
    diagnostics.main(sys.argv[1], sys.argv[2])
