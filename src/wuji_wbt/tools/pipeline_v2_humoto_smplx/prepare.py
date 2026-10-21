"""Fresh quick-finger seeds and contact patches from the fitted SMPL-X skin."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import time
import numpy as np
import trimesh
import mujoco
from config import *
sys.path.insert(0,str(RET/'retarget_pipeline'))
from pipeline_v2 import CoupledRefiner, Settings, initial_body

def main(seq):
    start=time.perf_counter();info=inputs(seq);z=np.load(info['canonical']);dest=OUT/'clips'/seq
    m=mujoco.MjModel.from_xml_path(str(info['model']));ma=mujoco.MjModel.from_xml_path(str(info['body_model']))
    body=initial_body(m,ma,np.load(info['body'])['qpos'])
    ref=CoupledRefiner(m,info['object'],Settings(**SETTINGS))
    begin=time.perf_counter()
    init=ref.quick_fingers(body,z['hand_kp_scaled'])
    np.savez_compressed(info['initialization'],qpos=init,fps=30)
    init_seconds=time.perf_counter()-begin
    write(dest/'run.json',dict(settings=SETTINGS,role='Fresh fitted SMPL-X input preparation'))
    E=module('smplx_source_metric_helpers',CODE.parent/'ref_candidates_metrics/evaluate.py')
    mesh=trimesh.load(info['object_mesh'],force='mesh');mesh.apply_scale(float(z['scale']))
    local=E.local(z['hand_kp_scaled'][:,:,E.TIPS],z['object_pose_mujoco'])
    _,distance=E.nearest(trimesh.proximity.ProximityQuery(mesh),local)
    human=z['human_joints_scaled'];toes=human[:,[E.SMPLH_DEMO_JOINTS.index(n) for n in ('L_Toe','R_Toe')]]
    support=np.zeros((len(human),2),bool)
    support[1:]=np.linalg.norm(np.diff(toes[:,:,:2],axis=0),axis=-1)<=.01
    support &= toes[:,:,2]<=np.min(toes[:,:,2])+.05
    info['masks'].parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(info['masks'],source_contact=distance<.03,source_support=support)
    # Renderer reads the historical cache path, containing labels only here.
    cache=OUT/'evaluation_v1/per_frame'/f'{seq}.npz';cache.parent.mkdir(parents=True,exist_ok=True)
    if not cache.exists():cache.symlink_to(info['masks'])
    begin=time.perf_counter()
    builder=module('smplx_source_targets',SHARED_CODE/'source_targets.py')
    original_closest=trimesh.proximity.closest_point
    def bounded_closest(mesh,points):
        # Closest-point queries are independent per point. Bound the temporary
        # candidate triangle arrays for complex meshes (notably working chair).
        if len(points)<=16:return original_closest(mesh,points)
        chunks=[original_closest(mesh,points[i:i+16]) for i in range(0,len(points),16)]
        return tuple(np.concatenate([chunk[k] for chunk in chunks]) for k in range(3))
    trimesh.proximity.closest_point=bounded_closest
    try:builder.prepare(seq)
    finally:trimesh.proximity.closest_point=original_closest
    audit=read(OUT/'targets'/f'{seq}.json')
    audit['source_masks_recomputed_from_smplx']=True
    audit['source_masks_unchanged_scope']='Within this SMPL-X run, relative to its prepared mask; not the previous native-HUMOTO mask.'
    audit['nearest_query_batch_size']=16
    audit['caveat']='Source skin is fitted SMPL-X geometry, not tactile ground truth. Patch/normal/gap formulas unchanged. Contact labels are recomputed from fitted SMPL-X landmarks; not reused native HUMOTO labels.'
    write(OUT/'targets'/f'{seq}.json',audit)
    write(dest/'prepared.json',dict(sequence=seq,frames=len(human),finger_initialization_seconds=init_seconds,
        targets_seconds=time.perf_counter()-begin,total_seconds=time.perf_counter()-start,
        fresh_initialization=True,fresh_targets=True,source_contact_finger_frames=int((distance<.03).sum())))

if __name__ == '__main__': run_stage('prepare',main,sys.argv[1])
