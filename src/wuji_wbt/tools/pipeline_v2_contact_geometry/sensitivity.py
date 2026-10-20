"""Post-hoc contact-threshold sensitivity; never replaces the frozen metrics.

The unchanged solver allows 1 mm penetration plus penalized slack. A contact
gate exactly at 1 mm is consequently sensitive to small numerical/soft-slack
changes. Report the complete declared threshold grid, not a selected cutoff.
"""
import csv
import numpy as np
from config import *

def main():
    rows=[]
    for seq in COHORT:
        for variant in REMOVALS:
            path=destination(seq,variant)/'contact_geometry_per_frame.npz'
            if not path.exists():continue
            z=np.load(path);source=z['source_contact'];skin=z['source_skin_contact']
            distance=z['tip_object_signed_distance_m'];gap=z['physical_patch_gap_m'];side=z['side_distance_m']
            for tolerance_mm in (.01,.1,.5,1.,1.5,2.,5.):
                predicted=(distance<=.002)&(distance>=-tolerance_mm/1000)
                tp=int((predicted&source).sum());fp=int((predicted&~source).sum());fn=int((~predicted&source).sum())
                intended=(gap<=.002)&(side>=-.001)&(distance>=-tolerance_mm/1000)
                rows.append(dict(sequence=seq,variant=variant,penetration_gate_mm=tolerance_mm,
                    gap_gate_mm=2.,source_contact_finger_frames=int(source.sum()),source_skin_contact_finger_frames=int(skin.sum()),
                    physical_f1_pct=200*tp/max(2*tp+fp+fn,1),
                    physical_recall_pct=100*tp/max(tp+fn,1),
                    intended_patch_recall_pct=float(intended[skin].mean()*100) if skin.any() else None))
    with (OUT/'contact_threshold_sensitivity.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(OUT/'contact_threshold_sensitivity.json',dict(post_hoc=True,
        reason='Inspect sensitivity to the unchanged 1 mm collision tolerance plus penalized slack. Primary contact metrics and existing definitions are unchanged.',
        thresholds_mm=[.01,.1,.5,1.,1.5,2.,5.],rows=rows))

if __name__=='__main__':main()
