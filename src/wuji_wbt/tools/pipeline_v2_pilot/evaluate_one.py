"""Matched evaluation: unchanged v1 definitions plus corrected penetration aggregation."""
from pathlib import Path
import importlib.util
import json
import sys
import time
import hashlib
import numpy as np
import mujoco
import trimesh
from scipy.spatial.transform import Rotation
from catalog import baseline_for

WORK = Path(__file__).resolve().parents[4]
ROOT = WORK.parent
BATCH = ROOT/'holosoma/eval_out/ref_candidates/batch_20261011_current_120'
OUT = ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_pilot_20261018'
spec = importlib.util.spec_from_file_location('frozen_eval', WORK/'src/wuji_wbt/tools/ref_candidates_metrics/evaluate.py')
E = importlib.util.module_from_spec(spec);spec.loader.exec_module(E)


def corrected(raw):
    obj = np.maximum(raw['hand_object_depth_m'].max(1), raw['body_object_depth'])
    g, s = raw['ground_depth'], raw['self_depth']
    mask = (obj > .02) | (g > .01) | (s > .01)
    depth = np.maximum.reduce([obj, g, s])
    return dict(combined_pen_duration_pct=float(mask.mean()*100),
                combined_pen_conditional_depth_cm=float(depth[mask].mean()*100) if mask.any() else None,
                penetration_violating_frames=int(mask.sum()),
                violating_frame_depth_sum_m=float(depth[mask].sum()))


def main(seq):
    start = time.perf_counter()
    BATCH = baseline_for(seq)
    job = E.read(BATCH/'jobs'/f'{seq}.json');obj = job['object']
    clip = BATCH/'clips'/seq;dest = OUT/'clips'/seq
    z = np.load(clip/'part0'/f'{seq}_canonical.npz', allow_pickle=False)
    Q = np.load(dest/'final_v2.npz')['qpos'];hk = z['hand_kp_scaled'];pose = z['object_pose_mujoco'];T = len(Q)
    assert T == len(hk)
    m = mujoco.MjModel.from_xml_path(job['model_xml'])
    # Check the actual unchanged body, including both root and object quaternions.
    qv1 = np.load(job['final_npz'])['qpos'];names = E.joints(m)
    selected = [a for name, (a, width) in names.items() if 'finger' in name or
                any(name == f'{side}_{joint}_joint' for side in ('left', 'right') for joint in
                    ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw'))]
    assert len(selected) == 54
    frozen = np.setdiff1d(np.arange(m.nq), selected)
    assert np.array_equal(Q[:, frozen], qv1[:, frozen])
    mesh = trimesh.load(E.RET/'models'/obj/f'{obj}.obj', force='mesh');mesh.apply_scale(float(z['scale']))
    pq = trimesh.proximity.ProximityQuery(mesh)
    local = E.local(hk[:, :, E.TIPS], pose);targets, distances = E.nearest(pq, local)
    with np.load(BATCH/'evaluation_v1/per_frame'/f'{seq}.npz') as cache:
        support = cache['source_support'];contact = cache['source_contact']
        assert np.array_equal(contact, distances < .03)
        before_raw = {k: cache['final_combined__'+k] for k in
                      ('hand_object_depth_m','body_object_depth','ground_depth','self_depth')}
    source = dict(hand_kp=hk, body_points=z['human_joints_scaled'][:, [E.SMPLH_DEMO_JOINTS.index(n) for n in E.LANDMARKS]],
                  local_tips=local, tip_distance=distances, surface_targets=targets,
                  contact_tip=contact, contact_hand=contact.any(-1), support_mask=support,
                  object_rotation=Rotation.from_quat(pose[:, [4,5,6,3]]).as_matrix())
    extracted = E.extract(m, Q, obj)
    # This comparison does not use a floating-hand target. The optional assembly-distance
    # diagnostic is undefined; every contact/alignment/body metric retains the original code.
    metrics, counts, raw = E.metrics(extracted, source, pq, pose, {'hand_kp': np.full_like(hk, np.nan)})
    metrics.pop('palm_to_floating_stageB_mm', None)
    for k in ('body_object_depth','ground_depth','self_depth','sole_height'):
        raw[k] = extracted[k]
    metrics.update(corrected(raw))
    old = E.read(BATCH/'evaluation_v1/per_sequence'/f'{seq}.json')['variant_metrics']['final_combined']
    old.update(corrected(before_raw));old.pop('palm_to_floating_stageB_mm', None)
    assert abs(metrics['foot_skating_mean_cm_s']-old['foot_skating_mean_cm_s']) < 1e-10
    fingers = [a for name, (a, _) in names.items() if 'finger' in name]
    init = np.load(dest/'finger_initialization.npz')['qpos']
    result = dict(sequence=seq, object=obj, frames=T, v1=old, v2=metrics, contact_counts=counts,
                  unchanged_body_exact=True, unchanged_source_contact_mask=True,
                  finger_change_from_quick_init_max_rad=float(np.abs(Q[:, fingers]-init[:, fingers]).max()),
                  evaluation_seconds=time.perf_counter()-start,
                  evaluator_sha256=hashlib.sha256(Path(E.__file__).read_bytes()).hexdigest())
    E.write(dest/'metrics.json', result)
    np.savez_compressed(dest/'per_frame_metrics.npz', **raw, source_contact=contact, source_support=support)
    print(json.dumps({k: result[k] for k in ('sequence','frames','evaluation_seconds','finger_change_from_quick_init_max_rad')}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
