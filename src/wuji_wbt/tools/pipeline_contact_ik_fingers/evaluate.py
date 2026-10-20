"""Frozen metrics plus separately labelled IK feasibility/torso-clearance diagnostics."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
import mujoco
import subprocess
from concurrent.futures import ThreadPoolExecutor
from config import *
from solver import ArmIK,targets,SETTINGS,TIPS


def supplemental(variant):
    info=inputs();dest=OUT/'variants'/variant/'clips'/SEQ
    Q=np.load(dest/'final_v2.npz')['qpos'];base=np.load(info['baseline'])['qpos']
    contact,surface=targets(info)
    m=mujoco.MjModel.from_xml_path(str(info['model']));ref=ArmIK(m,info['object'])
    clearance=[];penetration=[];frozen=[];lower=[];gap=[];violations=[];worst=[]
    for t,q in enumerate(Q):
        d,G,ids,active,v=ref.geometry(q)
        row=[]
        for side in ('left','right'):
            sidepair=[]
            for i in ids:
                names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or '' for g in ref.pairs[i,:2]]
                sidepair.append(ref.torso[i] and any(side in n for n in names))
            row.append(float(d[sidepair].min()) if any(sidepair) else SETTINGS['detect_distance'])
        clearance.append(row)
        penetration.append([float(max(0.,-d[ref.pairs[ids,2]==k].min())) if np.any(ref.pairs[ids,2]==k) else 0. for k in range(3)])
        violations.append(float(v[active].max()) if active.any() else 0.)
        frozen.append(float(v[~active].max()) if (~active).any() else 0.)
        ref.fk(q);p=ref.keypoints(False)
        per_lower=[];per_gap=[]
        for side in range(2):
            a=p[side,np.asarray(TIPS)[contact[t,side]]];b=surface[t,side,contact[t,side]]
            if not len(a):per_lower.append(np.nan);per_gap.append(np.nan);continue
            ac=a-a.mean(0);bc=b-b.mean(0);u,_,vt=np.linalg.svd(ac.T@bc)
            correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@vt)
            rotation=u@correction@vt
            per_lower.append(float(np.sqrt(np.mean(np.sum((ac@rotation-bc)**2,axis=1)))))
            per_gap.append(float(np.sqrt(np.mean(np.sum((a-b)**2,axis=1)))))
        lower.append(per_lower);gap.append(per_gap)
        if active.any() and v[active].max()>1e-5:
            i=np.flatnonzero(active)[np.argmax(v[active])];a,b,k=ref.pairs[ids[i]]
            worst.append(dict(frame=t,pair_id=int(ids[i]),kind=int(k),distance_mm=float(d[i]*1000),
                required_margin_mm=float(ref.margin[ids[i]]*1000),
                bodies=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) for g in (a,b)]))
    clearance=np.asarray(clearance);penetration=np.asarray(penetration);violations=np.asarray(violations)
    lower=np.asarray(lower);gap=np.asarray(gap);mask=contact.any(-1)
    armcols=ref.arms;armadr=ref.qadr[armcols]
    arms=Q[:,armadr].reshape(-1,2,7);lo=ref.lo[armcols].reshape(2,7);hi=ref.hi[armcols].reshape(2,7)
    margins=np.minimum(arms-lo,hi-arms).min(-1)
    result=dict(variant=variant,frames=len(Q),collision_backend=SETTINGS['collision_backend'],
        collision_feasible_frames=int(np.sum(violations<=SETTINGS['feasibility_tolerance'])),
        frozen_collision_frames=int(np.sum(np.asarray(frozen)>SETTINGS['feasibility_tolerance'])),
        left_torso_min_clearance_mm=float(clearance[:,0].min()*1000),right_torso_min_clearance_mm=float(clearance[:,1].min()*1000),
        left_torso_contact_mean_clearance_mm=float(clearance[mask[:,0],0].mean()*1000),
        torso_clearance_below5mm_hand_frames_pct=float(np.mean(clearance[mask]<.005-1e-5)*100),
        object_max_depth_mm=float(penetration[:,0].max()*1000),ground_max_depth_mm=float(penetration[:,1].max()*1000),
        self_max_depth_mm=float(penetration[:,2].max()*1000),
        contact_point_rms_mm=float(np.sqrt(np.nanmean(gap**2))*1000),
        fixed_finger_rigid_fit_lower_bound_rms_mm=float(np.sqrt(np.nanmean(lower**2))*1000),
        contact_hand_frames_rigid_lower_bound_gt5mm_pct=float(np.mean(lower[mask]>.005)*100),
        arm_min_margin_contact_mean_deg=float(np.degrees(margins[mask]).mean()),
        arm_min_margin_contact_min_deg=float(np.degrees(margins[mask]).min()),
        arm_contact_near_limit5deg_pct=float(np.mean(margins[mask]<np.deg2rad(5))*100),
        arm_change_rms_deg=float(np.degrees(np.sqrt(np.mean((Q[:,armadr]-base[:,armadr])**2)))),
        frozen_coordinates_exact=bool(np.array_equal(Q[:,np.setdiff1d(np.arange(m.nq),ref.qadr)],base[:,np.setdiff1d(np.arange(m.nq),ref.qadr)])),
        worst_controllable_pairs=worst,
        definitions=dict(torso='Minimum independent Coal convex distance among registered arm/hand-to-torso/pelvis/waist pairs, separately per side, capped at 40 mm. Same allowed-pair filters as baseline.',
            feasibility='Zero object/ground/other-self margin and 5 mm torso margin, tolerance 10 micrometres; zero active-joint Jacobian rows reported separately.',
            rigid_lower_bound='Per contacting hand-frame, optimal unconstrained proper rigid alignment of current robot fingertip centers to source surface anchors. Omits collisions and arm reach: optimistic lower bound on the current hand configuration; with adjustable fingers this is not a bound on the full optimization.',
            smoothness='30 Hz finite differences, no filtering; derivative stencils wholly within the source-contact interval.'))
    for order,key in ((1,'velocity'),(2,'acceleration'),(3,'jerk')):
        values=np.diff(arms,n=order,axis=0)*30**order
        valid=np.logical_and.reduce([mask[i:len(mask)-order+i] for i in range(order+1)])
        result[f'arm_contact_{key}_rms']=float(np.sqrt(np.mean(values[valid]**2)))
    fingers=Q[:,ref.qadr[ref.fingers]]
    result['finger_change_rms_deg']=float(np.degrees(np.sqrt(np.mean((fingers-base[:,ref.qadr[ref.fingers]])**2))))
    result['finger_change_max_deg']=float(np.degrees(np.abs(fingers-base[:,ref.qadr[ref.fingers]]).max()))
    result['finger_acceleration_rms']=float(np.sqrt(np.mean((np.diff(fingers,n=2,axis=0)*30**2)**2)))
    # Compare identical registered collision pairs for all trajectories, including zero-gradient rows.
    for k,label in enumerate(('object','ground','self')):
        for mm in (.01,.1,1,5,10,20):
            result[f'coal_{label}_duration_above_{mm:g}mm_pct']=float(np.mean(penetration[:,k]>mm/1000)*100)
    result['all_registered_clearances_feasible_frames']=int(np.sum((violations<=SETTINGS['feasibility_tolerance'])&(np.asarray(frozen)<=SETTINGS['feasibility_tolerance'])))
    np.savez_compressed(dest/'ik_diagnostics_per_frame.npz',torso_clearance_m=clearance,
        max_depth_object_ground_self_m=penetration,controllable_violation_m=violations,
        frozen_violation_m=frozen,rigid_contact_lower_bound_m=lower,contact_rms_m=gap)
    write(dest/'ik_diagnostics.json',result)
    return result


def main():
    pilot=module('contact_ik_frozen_evaluator',PILOT_CODE/'evaluate_one.py')
    pilot.OUT=OUT/'variants/contact_ik';pilot.main(SEQ)
    def worker(variant):
        subprocess.run([PY,__file__,'--variant',variant],check=True)
        return read(OUT/'variants'/variant/'clips'/SEQ/'ik_diagnostics.json')
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=dict(zip(REMOVALS,pool.map(worker,REMOVALS)))
    assert all(r['frozen_coordinates_exact'] for r in results.values())
    write(OUT/'diagnostic_summary.json',results)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=list(REMOVALS))
    args=parser.parse_args()
    if args.variant:supplemental(args.variant)
    else:main()
