"""Matched reported metrics plus an explicit all-body collision audit."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import csv
import numpy as np
import mujoco
from config import *
import solver
from holosoma_retargeting.retarget_pipeline.object_colliders import validate_body_object_colliders

def summarize(depth):
    return dict(peak_mm=float(depth.max()*1000),mean_frame_max_mm=float(depth.mean()*1000),
        peak_frame=int(depth.argmax()),duration_over1mm_pct=float(np.mean(depth>.001)*100),
        duration_over10mm_pct=float(np.mean(depth>.01)*100),duration_over20mm_pct=float(np.mean(depth>.02)*100))

def audit(ref,Q,obj):
    m=ref.m;names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or
        (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or '') for g in range(m.ngeom)]
    active=np.flatnonzero((m.geom_contype!=0)|(m.geom_conaffinity!=0))
    objects=[g for g in active if names[g].startswith(obj+'_piece')]
    planes=[g for g in active if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE]
    robot=[g for g in active if g not in objects+planes]
    moving=set(ref.pairs[ref.pairs[:,2]==0,0]);legs={g for g in robot if any(w in names[g] for w in ('hip','knee','ankle'))}
    pairs=np.array([(a,b) for a in robot for b in objects]);a,b=pairs.T
    result=np.zeros((len(Q),4));worst=None
    for t,q in enumerate(Q):
        ref.fk(q)
        lb=np.linalg.norm(ref.d.geom_xpos[a]-ref.d.geom_xpos[b],axis=1)-m.geom_rbound[a]-m.geom_rbound[b]
        for i in np.flatnonzero(lb<0):
            dist,_,_,_=ref.queries.query(ref.d,int(a[i]),int(b[i]))
            depth=max(0,-dist)
            if depth>result[t,0]:
                result[t,0]=depth
                if worst is None or depth>worst['depth_m']:
                    worst=dict(frame=t,time_s=t/30,geometry_a=names[a[i]],geometry_b=names[b[i]],depth_m=float(depth))
            if a[i] in legs:result[t,1]=max(result[t,1],depth)
            if a[i] not in moving:result[t,2]=max(result[t,2],depth)
        for g in robot:
            for p in planes:
                dist=ref.queries.query(ref.d,g,p)[0];result[t,3]=max(result[t,3],-dist)
    return dict(all_robot_object=summarize(result[:,0]),legs_object=summarize(result[:,1]),
        fixed_body_object=summarize(result[:,2]),all_robot_ground=summarize(result[:,3]),worst_pair=worst),result

def main(seq):
    info,z,data,ref,body,init=solver.load(seq,'mujoco');result={};arrays={}
    for label,root in [('before',PREVIOUS),('after',OUT)]:
        dest=root/'variants/mujoco/clips'/seq;Q=np.load(dest/'final_v2.npz')['qpos']
        measures,perframe=audit(ref,Q,info['object']);arrays[label]=perframe
        metrics=read(dest/'metrics.json')['v2'];contact=read(dest/'diagnostics.json')
        result[label]=dict(collision=measures,physical_contact_f1_pct=contact['physical_touch_vs_original_source']['f1_pct'],
            physical_contact_recall_pct=contact['physical_touch_vs_original_source']['recall_pct'],
            strict_physical_contact_f1_pct=contact['strict_physical_touch_vs_original_source']['f1_pct'],
            intended_patch_gap_mm=contact['physical_patch_gap_mean_mm'],
            foot_skating_cm_s=metrics['foot_skating_mean_cm_s'],
            original_metrics=metrics)
    # Preserve exact inputs and hand settings; assert only root/body output may differ.
    previous=np.load(PREVIOUS/'clips'/seq/'part0'/f'{seq}_canonical.npz')
    for key in ('human_joints_scaled','hand_kp_scaled','object_pose_mujoco','object_points_scaled'):
        np.testing.assert_array_equal(z[key],previous[key])
    Q=np.load(destination(seq)/'final_v2.npz')['qpos'];oldQ=np.load(PREVIOUS/'variants/mujoco/clips'/seq/'final_v2.npz')['qpos']
    np.testing.assert_array_equal(Q[:,-7:],oldQ[:,-7:])
    for k,v in data.items():
        original=np.load(PREVIOUS/'targets'/f'{seq}.npz')[k]
        np.testing.assert_array_equal(v,original)
    fixed=np.setdiff1d(np.arange(ref.m.nq),ref.qadr)
    np.testing.assert_array_equal(Q[:,fixed],body[:,fixed])
    coverage=validate_body_object_colliders(mujoco.MjModel.from_xml_path(str(info['body_model'])),info['object'])
    baseline_run=read(PREVIOUS/'variants/mujoco/clips'/seq/'run.json')
    run=read(destination(seq)/'run.json')
    for key in ('settings','extra','options'):assert run[key]==baseline_run[key],key
    result.update(sequence=seq,frames=len(Q),body_collision_coverage=coverage,
        definitions='Historical evaluation formulas/thresholds unchanged. Supplemental penetration is maximum signed Coal depth across ALL active robot-object pairs, including legs; ground uses analytic plane queries. Mean includes all frames.',
        inputs_targets_object_trajectory_exact=True,hand_settings_unchanged=True,
        body_seconds=read(marker(seq,'body'))['seconds'],refinement_seconds=run['joint_refinement_seconds'],
        videos=read(marker(seq,'render'))['video_validation'])
    result['experiment_code_sha256']={p.name:sha(p) for p in CODE.glob('*.py')}
    result['body_pipeline_sha256']={name:sha(RET/'retarget_pipeline'/name)
                                   for name in ('pipeline.py','object_colliders.py')}
    np.savez_compressed(OUT/'full_body_collision_per_frame.npz',**arrays)
    write(OUT/'comparison.json',result)
    fields=[('Leg/object peak penetration (mm)','collision.legs_object.peak_mm'),
        ('Leg/object mean frame-max penetration (mm)','collision.legs_object.mean_frame_max_mm'),
        ('Leg/object frames above 10 mm (%)','collision.legs_object.duration_over10mm_pct'),
        ('Whole robot/object peak penetration (mm)','collision.all_robot_object.peak_mm'),
        ('Whole robot/object mean frame-max penetration (mm)','collision.all_robot_object.mean_frame_max_mm'),
        ('Whole robot/ground peak penetration (mm)','collision.all_robot_ground.peak_mm'),
        ('Physical contact recall (%)','physical_contact_recall_pct'),('Physical contact F1 (%)','physical_contact_f1_pct'),
        ('Mean intended-patch gap (mm)','intended_patch_gap_mm'),('Foot skating (cm/s)','foot_skating_cm_s')]
    def get(label,path):
        v=result[label]
        for key in path.split('.'):v=v[key]
        return v
    rows=[dict(metric=label,before=get('before',key),after=get('after',key)) for label,key in fields]
    with (OUT/'comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['metric','before','after']);writer.writeheader();writer.writerows(rows)
    for row in rows:print(row,flush=True)

if __name__=='__main__':main(sys.argv[1])
