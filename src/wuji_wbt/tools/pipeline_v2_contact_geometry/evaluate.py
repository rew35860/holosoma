"""Historical evaluator untouched; additional contact-geometry diagnostics separate."""
import numpy as np
from config import *
from solver import load,target_at

def scores(actual,predicted):
    tp=int(np.sum(actual&predicted));fp=int(np.sum(~actual&predicted));fn=int(np.sum(actual&~predicted))
    return dict(tp=tp,fp=fp,fn=fn,precision_pct=100*tp/max(tp+fp,1),
        recall_pct=100*tp/max(tp+fn,1),f1_pct=200*tp/max(2*tp+fp+fn,1))

def main(seq,variant):
    frozen=module('contact_geometry_frozen_evaluator',PILOT_CODE/'evaluate_one.py')
    frozen.OUT=OUT/'variants'/variant;frozen.main(seq)
    info,z,data,ref,body,init=load(seq,'both');dest=destination(seq,variant)
    Q=np.load(dest/'final_v2.npz')['qpos'];T=len(Q);mask=data['contact']
    gap=np.full((T,2,5),np.nan);side=np.full_like(gap,np.nan);witness=np.full((T,2,5,3),np.nan)
    depth=np.zeros((T,3));tip_distance=np.full((T,2,5),ref.cfg.collision_detect)
    for t,q in enumerate(Q):
        target=target_at(ref,z,data,t);ref.fk(q);p=ref.keypoints(False)
        for s,f in np.argwhere(mask[t]):
            s,f=int(s),int(f)
            value,_,_,pb=ref.gap_query(target,s,f,False);gap[t,s,f]=value
            local=(p[s,[4,8,12,16,20][f]]-target['object_pose'][:3])@target['R']
            side[t,s,f]=data['direction_local'][t,s,f]@(local-data['anchor_local'][t,s,f])
            witness[t,s,f]=(pb-target['object_pose'][:3])@target['R']
        distances,_,ids=ref.collisions(q,False)
        for k in range(3):
            selected=ref.pairs[ids,2]==k
            if selected.any():depth[t,k]=max(0.,-float(distances[selected].min()))
        for s in range(2):
            for f,g in enumerate(ref.tipgeom[s]):
                selected=(ref.pairs[ids,2]==0)&(ref.pairs[ids,0]==g)
                if selected.any():tip_distance[t,s,f]=float(distances[selected].min())
    # Original 30-mm source mask remains the primary label. Stricter source-skin
    # contact is a separately reported diagnostic, never substituted silently.
    physical=(tip_distance<=.002)&(tip_distance>=-.001)
    strict=(tip_distance<=.002)&(tip_distance>=-1e-5)
    source_skin_contact=mask&(data['source_skin_gap']<=.002)
    intended=(gap<=.002)&(side>=-.001)&(tip_distance>=-.001)
    arms=np.concatenate(ref.arm_columns);armQ=Q[:,ref.qadr[arms]].reshape(T,2,7)
    margin=np.minimum(Q[:,ref.qadr]-ref.lo,ref.hi-Q[:,ref.qadr])
    arm_margin=margin[:,arms].reshape(T,2,7).min(-1);hand_contact=mask.any(-1)
    frozen_ids=np.setdiff1d(np.arange(ref.m.nq),ref.qadr)
    assert np.array_equal(Q[:,frozen_ids],body[:,frozen_ids])
    assert np.all(margin>=-1e-7)
    rate=float(np.abs(np.diff(Q[:,ref.qadr],axis=0)).max())
    assert rate<=ref.cfg.joint_rate_limit+1e-6
    relative=witness-data['anchor_local'];valid=mask[1:]&mask[:-1]
    drift=np.linalg.norm(np.diff(relative,axis=0),axis=-1)*30*100
    result=dict(sequence=seq,variant=variant,frames=T,
        correct_side_contact_pct=float(np.mean(side[mask]>=-.001)*100),
        side_violation_mean_mm=float(np.maximum(-side[mask],0).mean()*1000),
        side_violation_max_mm=float(np.maximum(-side[mask],0).max()*1000),
        physical_patch_gap_mean_mm=float(gap[mask].mean()*1000),
        source_skin_gap_mean_mm=float(data['source_skin_gap'][mask].mean()*1000),
        surface_gap_target_mae_mm=float(np.abs(gap[mask]-data['source_skin_gap'][mask]).mean()*1000),
        intended_patch_touch_recall_skin_contact_pct=float(intended[source_skin_contact].mean()*100) if source_skin_contact.any() else None,
        physical_touch_vs_original_source=scores(mask,physical),
        strict_physical_touch_vs_original_source=scores(mask,strict),
        physical_touch_vs_source_skin=scores(source_skin_contact,physical),
        relative_surface_witness_drift_cm_s=float(drift[valid].mean()),
        arm_contact_near_limit5deg_pct=float(np.mean(arm_margin[hand_contact]<np.deg2rad(5))*100),
        arm_contact_min_margin_mean_deg=float(np.degrees(arm_margin[hand_contact]).mean()),
        arm_acceleration_rms_rad_s2=float(np.sqrt(np.mean((np.diff(armQ,n=2,axis=0)*30**2)**2))),
        arm_jerk_rms_rad_s3=float(np.sqrt(np.mean((np.diff(armQ,n=3,axis=0)*30**3)**2))),
        finger_acceleration_rms_rad_s2=float(np.sqrt(np.mean((np.diff(Q[:,ref.qadr[np.concatenate(ref.finger_columns)]],n=2,axis=0)*30**2)**2))),
        frozen_body_object_exact=True,max_active_step_rad=rate,
        definitions=dict(side='Existing geometry reference point in intended source exterior halfspace; 1 mm tolerance. This does not measure pad orientation.',
            physical_touch='Tip collision convex distance to all object collision pieces <=2 mm, >=-1 mm. Strict version >=-10 micrometres.',
            intended_touch='Original-mesh source patch gap <=2 mm, reference point on correct side within 1 mm, signed tip-object material penetration <=1 mm. Recall only on source skin gap <=2 mm.',
            drift='Mean speed of object-frame patch witness minus current source anchor, using adjacent source-contact frames; includes surface rolling and witness switching.',
            source_skin='Source-only reconstructed skin ROI within 20 mm of each source landmark, nearest-tip-assigned. This label is supplemental, not the frozen 30-mm mask.',
            penetration='Same registered V2 object/ground/self collision pairs; Coal signed distances, per-frame maximum. Frozen historical evaluator retained separately.'))
    for k,label in enumerate(('object','ground','self')):
        result[f'{label}_max_penetration_mm']=float(depth[:,k].max()*1000)
        result[f'{label}_mean_frame_max_penetration_mm']=float(depth[:,k].mean()*1000)
        for threshold in (.01,.1,1,5,10,20):
            result[f'{label}_penetration_above_{threshold:g}mm_pct']=float(np.mean(depth[:,k]>threshold/1000)*100)
    write(dest/'diagnostics.json',result)
    np.savez_compressed(dest/'contact_geometry_per_frame.npz',source_contact=mask,source_skin_contact=source_skin_contact,
        physical_patch_gap_m=gap,source_skin_gap_m=data['source_skin_gap'],side_distance_m=side,
        object_witness_local=witness,tip_object_signed_distance_m=tip_distance,max_depth_object_ground_self_m=depth)

if __name__=='__main__':main(*sys.argv[1:])
