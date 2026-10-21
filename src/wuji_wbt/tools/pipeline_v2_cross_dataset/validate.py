"""Independent artifact checks and frozen provenance for the completed pilot."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import importlib.metadata,subprocess
import numpy as np
import mujoco
from config import *

def main():
    checks=[];inputs_sha={}
    for r in rows():
        seq=r['sequence'];info=inputs(seq);z=np.load(info['canonical']);m=mujoco.MjModel.from_xml_path(str(info['model']))
        for key in ('canonical','body','body_model','model','human','initialization','masks','object_mesh'):
            inputs_sha[str(info[key])]=sha(info[key])
        inputs_sha[str(OUT/'targets'/f'{seq}.npz')]=sha(OUT/'targets'/f'{seq}.npz')
        for variant in VARIANTS:
            folder=OUT/'variants'/variant/'clips'/seq
            if not (folder/'evaluate_completed.json').exists():continue
            Q=np.load(folder/'final_v2.npz')['qpos'];run=read(folder/'run.json')
            assert len(Q)==len(z['hand_kp_scaled']) and Q.shape[1]==m.nq and np.isfinite(Q).all()
            np.testing.assert_allclose(Q[:,-7:],z['object_pose_mujoco'],rtol=0,atol=1e-9)
            for sl in (slice(3,7),slice(-4,None)):
                np.testing.assert_allclose(np.linalg.norm(Q[:,sl],axis=1),1.,rtol=0,atol=1e-6)
            ids=np.flatnonzero(m.jnt_limited);q=Q[:,m.jnt_qposadr[ids]]
            violation=float(max(0.,np.max(m.jnt_range[ids,0]-q),np.max(q-m.jnt_range[ids,1])))
            assert violation<=1e-5
            assert run['evaluation_targets_sha256']==sha(OUT/'targets'/f'{seq}.npz')
            assert run['settings']['contact_weight']==0 and run['signed_recovery'] is False
            assert run['settings']['fingertip_weight']==(6.25 if variant=='tip_relaxed' else 25.)
            assert run['extra']['gap_weight']==(1024. if variant=='surface_emphasis' else 256.)
            checks.append(dict(sequence=seq,variant=variant,frames=len(Q),passed=True,
                all_robot_joint_bound_violation_rad=violation,object_trajectory_exact=True,finite=True,
                unchanged_target_sha256=run['evaluation_targets_sha256']))
    sources=set(CODE.glob('*.py'))
    for directory in [BACKEND_CODE,IMPLEMENTATION_CODE,GEOMETRY_CODE,BASE_CODE,SIGNED_CODE,SURFACE_CODE,SHARED_CODE,
                      CODE.parent/'pipeline_v2_humoto_smplx',CODE.parent/'pipeline_v2_proxy_motion',
                      CODE.parent/'ref_candidates_metrics',CODE.parent/'pipeline_contact_ik',RET/'retarget_pipeline']:
        sources.update(directory.glob('*.py'))
    sources.update([RET/'src/interaction_mesh_retargeter.py',RET/'wuji/topoRetarget.py'])
    versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','trimesh','mujoco','clarabel','coacd','smplx','torch')}
    write(OUT/'validation.json',dict(passed=True,checks=checks,completed_runs=len(checks),
        interpretation='Structural, numerical and provenance checks; not a physical grasp-success certificate.'))
    write(OUT/'provenance.json',dict(git_branch=subprocess.check_output(['git','branch','--show-current'],cwd=WORK,text=True).strip(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),
        versions=versions,code_sha256={str(p):sha(p) for p in sorted(sources)},input_sha256=inputs_sha,
        learned_models_redistributed=False,source='Workspace files and cached licensed/source dataset assets'))
    for path in sources:
        dest=OUT/'source_snapshot'/path.relative_to(WORK);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    print('Validated',len(checks),'motion artifacts',flush=True)
if __name__=='__main__':main()
