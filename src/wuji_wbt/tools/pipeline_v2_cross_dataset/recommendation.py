"""Do not promote a tuned weight unless paired validation supports it."""
import numpy as np
from config import *

def main():
    selected=read(OUT/'selection.json')['selected'];records=read(OUT/'per_sequence.json')
    groups=[];passed=selected!='current'
    for dataset in ('humoto','hiphi'):
        baseline={r['sequence']:r for r in records if r['dataset']==dataset and r['split']=='validation' and r['variant']=='current'}
        trial={r['sequence']:r for r in records if r['dataset']==dataset and r['split']=='validation' and r['variant']==selected}
        common=sorted(baseline.keys()&trial.keys());checks={};means={}
        if common:
            keys=['patch_gap_mm','physical_f1_pct','intended_contact_pct','hand_object_peak_mm','self_peak_mm','arm_jerk_rad_s3','near_arm_limit_pct']
            means={label:{k:float(np.mean([data[s][k] for s in common])) for k in keys} for label,data in [('current',baseline),('candidate',trial)]}
            a,b=means['current'],means['candidate']
            checks=dict(no_lost_solves=len(trial)>=len(baseline),gap_improves=b['patch_gap_mm']<a['patch_gap_mm'],
                physical_f1_preserved=b['physical_f1_pct']>=a['physical_f1_pct']-2,
                intended_contact_preserved=b['intended_contact_pct']>=a['intended_contact_pct']-2,
                object_penetration_preserved=b['hand_object_peak_mm']<=a['hand_object_peak_mm']+1,
                self_penetration_preserved=b['self_peak_mm']<=a['self_peak_mm']+1,
                smoothness_preserved=b['arm_jerk_rad_s3']<=a['arm_jerk_rad_s3']*1.25,
                joint_margin_preserved=b['near_arm_limit_pct']<=a['near_arm_limit_pct']+5)
        passed &= bool(checks) and all(checks.values())
        groups.append(dict(dataset=dataset,baseline_successes=len(baseline),candidate_successes=len(trial),paired=len(common),means=means,checks=checks))
    recommended=selected if passed else 'current'
    write(OUT/'recommendation.json',dict(tuned_candidate=selected,recommended=recommended,
        validation_passed=bool(passed),per_dataset=groups,
        policy='Require improvement without defined regressions separately on HUMOTO and HiPHI. Otherwise retain existing weights. This is a small-window pilot, not universal validation.'))
    write(OUT/'recommended_preset.json',dict(variant=recommended,pipeline_version='v3' if recommended=='surface_emphasis' else 'v2',
        overrides={'current':{},'tip_relaxed':{'fingertip_weight':6.25},'surface_emphasis':{'gap_weight':1024.}}[recommended],
        common=dict(geometry_side=True,signed_recovery=False,contact_weight=0.,collision_backend='mujoco',
            collision_aware_frame0=True,motion_compensated_later_frames=True,body_object_collision=True),
        scope='Use with the same tested V2 implementation and source-target builder; model/source checks remain mandatory.'))
    return recommended
if __name__=='__main__':main()
