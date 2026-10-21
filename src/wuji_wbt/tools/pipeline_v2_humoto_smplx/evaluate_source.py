"""Frozen metric formulas, freshly computed SMPL-X source labels and body checks."""
import time
import numpy as np
import mujoco
import trimesh
from scipy.spatial.transform import Rotation
from config import *

def main(seq):
    start=time.perf_counter();info=inputs(seq);dest=destination(seq)
    E=module('smplx_frozen_metric_functions',CODE.parent/'ref_candidates_metrics/evaluate.py')
    legacy=_corrected_module()
    z=np.load(info['canonical']);Q=np.load(dest/'final_v2.npz')['qpos']
    hk=z['hand_kp_scaled'];pose=z['object_pose_mujoco']
    m=mujoco.MjModel.from_xml_path(str(info['model']))
    sys.path.insert(0,str(RET/'retarget_pipeline'))
    from pipeline_v2 import initial_body
    body=initial_body(m,mujoco.MjModel.from_xml_path(str(info['body_model'])),np.load(info['body'])['qpos'])
    names=E.joints(m)
    selected=[a for n,(a,w) in names.items() if 'finger' in n or any(n==f'{s}_{j}_joint'
        for s in ('left','right') for j in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw'))]
    assert len(selected)==54
    fixed=np.setdiff1d(np.arange(m.nq),selected)
    assert np.array_equal(Q[:,fixed],body[:,fixed])
    mesh=trimesh.load(info['object_mesh'],force='mesh');mesh.apply_scale(float(z['scale']))
    pq=trimesh.proximity.ProximityQuery(mesh)
    local=E.local(hk[:,:,E.TIPS],pose);targets,distances=E.nearest(pq,local)
    masks=np.load(info['masks']);contact=masks['source_contact'];support=masks['source_support']
    assert np.array_equal(contact,distances<.03)
    source=dict(hand_kp=hk,body_points=z['human_joints_scaled'][:,[E.SMPLH_DEMO_JOINTS.index(n) for n in E.LANDMARKS]],
        local_tips=local,tip_distance=distances,surface_targets=targets,contact_tip=contact,
        contact_hand=contact.any(-1),support_mask=support,object_rotation=Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix())
    extracted=E.extract(m,Q,info['object'])
    metrics,counts,raw=E.metrics(extracted,source,pq,pose,{'hand_kp':np.full_like(hk,np.nan)})
    metrics.pop('palm_to_floating_stageB_mm',None)
    for k in ('body_object_depth','ground_depth','self_depth','sole_height'):raw[k]=extracted[k]
    metrics.update(legacy.corrected(raw))
    fingers=[a for n,(a,w) in names.items() if 'finger' in n]
    init=np.load(info['initialization'])['qpos']
    result=dict(sequence=seq,object=info['object'],frames=len(Q),v2=metrics,contact_counts=counts,
        unchanged_body_exact=True,source='fitted SMPL-X',source_masks_recomputed=True,
        finger_change_from_quick_init_max_rad=float(np.abs(Q[:,fingers]-init[:,fingers]).max()),
        evaluation_seconds=time.perf_counter()-start,evaluator_sha256=sha(E.__file__),
        definitions='Same metric functions and thresholds as native-HUMOTO run. New SMPL-X source labels; do not interpret differences as a fixed-label ablation.')
    write(dest/'metrics.json',result)
    np.savez_compressed(dest/'per_frame_metrics.npz',**raw,source_contact=contact,source_support=support)

def _corrected_module():
    # Direct loader bypasses the wrapper routing; only its pure aggregation is used.
    import importlib.util
    spec=importlib.util.spec_from_file_location('smplx_original_aggregation',PILOT_CODE/'evaluate_one.py')
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj
