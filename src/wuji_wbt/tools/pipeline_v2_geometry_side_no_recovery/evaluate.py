"""Frozen historical/contact metrics, plus separately named geometry-side checks."""
import numpy as np
from config import *
from solver import load,target_at


def main(seq,variant):
    module('no_recovery_frozen_evaluator',BASE_CODE/'evaluate.py').main(seq,variant)
    _,z,data,ref,_,_=load(seq,variant)
    dest=destination(seq,variant);Q=np.load(dest/'final_v2.npz')['qpos']
    support=np.full(data['contact'].shape,np.nan)
    for t,q in enumerate(Q):
        target=target_at(ref,z,data,t);ref.fk(q)
        for side,finger in np.argwhere(data['contact'][t]):
            support[t,side,finger]=ref.support_side(target,int(side),int(finger),False)[0]
    cache=np.load(dest/'contact_geometry_per_frame.npz');mask=cache['source_contact']
    signed=cache['tip_object_signed_distance_m'];gap=cache['physical_patch_gap_m']
    depths=np.maximum(-signed[mask],0.)*1000
    intended=(support>=-.001)&(gap<=.002)&(signed>=-.001)
    stats=dict(geometry_correct_side_pct=float(100*np.mean(support[mask]>=-.001)),
        geometry_wrong_side_mean_mm=float(np.maximum(-support[mask]-.001,0.).mean()*1000),
        geometry_intended_contact_pct=float(100*np.mean(intended[mask])),
        contact_tip_mean_pen_mm=float(depths.mean()),contact_tip_p95_pen_mm=float(np.percentile(depths,95)),
        contact_tip_max_pen_mm=float(depths.max()),contact_finger_frames=int(mask.sum()),
        definitions=dict(geometry_side='Minimum distal-tip collision vertex plane coordinate >= -1 mm.',
            geometry_intended_contact='Geometry-side clearance >= -1 mm, intended patch gap <=2 mm, signed tip material penetration <=1 mm; all original source-contact fingertip-frames.',
            contact_tip_penetration='max(-signed distal-tip/object distance,0); all original source-contact fingertip-frames, including zeros. Supplemental, not a replacement of frozen metrics.'))
    for threshold in (.01,.1,.5,1.,1.5,2.,5.):
        stats[f'contact_tip_pen_above_{threshold:g}mm_pct']=float(100*np.mean(depths>threshold))
    per_digit=[]
    for side in range(2):
        for finger in range(5):
            valid=mask[:,side,finger];n=int(valid.sum())
            if not n:continue
            per_digit.append(dict(side=('left','right')[side],finger=('thumb','index','middle','ring','little')[finger],
                frames=n,reference_side_pct=float(100*np.mean(cache['side_distance_m'][:,side,finger][valid]>=-.001)),
                geometry_side_pct=float(100*np.mean(support[:,side,finger][valid]>=-.001)),
                intended_geometry_contact_pct=float(100*np.mean(intended[:,side,finger][valid])),
                mean_patch_gap_mm=float(gap[:,side,finger][valid].mean()*1000),
                mean_pen_mm=float(np.maximum(-signed[:,side,finger][valid],0.).mean()*1000),
                max_pen_mm=float(np.maximum(-signed[:,side,finger][valid],0.).max()*1000)))
    stats['per_digit']=per_digit
    write(dest/'geometry_diagnostics.json',stats)
    np.savez_compressed(dest/'geometry_side_per_frame.npz',support_side_m=support,
                        geometry_intended_contact=intended,source_contact=mask)


if __name__=='__main__':main(*sys.argv[1:])
