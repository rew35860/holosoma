"""Freeze source-only cohort before solving, with no outcome-based exclusion."""
from config import *
def main():
    assert not (OUT/'cohort.json').exists()
    native=[('push_table_forward_with_both_hands-039','tune'),
            ('lifting_and_checking_step_stool-463','tune'),
            ('pick_up_and_carry_organizer_medium_while_walking-096','validation'),
            ('drag_utility_cart_on_ground_while_walking_back_and_forth-677','validation'),
            ('carrying_working_chair_with_right_hand-839','validation'),
            ('shifting_shelf_with_left_leg-241','validation'),
            ('drag_dining_chair_on_ground_while_walking-444','validation'),
            ('picking_up_the_woven_basket_from_handle_and_walking_from_left_to_right-160','validation')]
    result=[]
    for seq,split in native:
        source=NATIVE if (NATIVE/'jobs'/f'{seq}.json').exists() else BATCH
        job=read(source/'jobs'/f'{seq}.json')
        result.append(dict(sequence=seq,dataset='humoto',object=job['object'],split=split,
            source=str(source),source_start_frame=0,maximum_frames=240,
            previously_used_in_v2=source==NATIVE))
    hiphi=[('Placing-put_0030','tune'),('Filling-douse_0014','tune'),
           ('Bringing-carry_0434','validation'),('Cause_motion-move_0414','validation'),
           ('Manipulation-squeeze_0005','validation'),('Cause_motion-shove_0066','validation')]
    catalog={r['sequence']:r for r in read(HIPHI/'cohort.json')}
    for seq,split in hiphi:
        r=catalog[seq]
        result.append(dict(sequence=seq,dataset='hiphi',object=r['mesh_id'].lower(),split=split,
            source=str(HIPHI/'pose_plus_shape'/f'{seq}.npz'),metadata=r,
            source_start_frame=0,maximum_frames=240,previously_used_in_v2=False))
    OUT.mkdir(parents=True,exist_ok=True);write(OUT/'cohort.json',result)
    write(OUT/'protocol.json',dict(predeclared=True,frames='First 240 source frames at 30 FPS; 8-second pilot, not full sequences',
        datasets={'humoto':'Original Mixamo/native skin, corrected shared floor; no SMPL fit',
                  'hiphi':'Existing hiphi2smplx pose-plus-shape fit with articulated hands; CPU forward evaluation'},
        splits='4 tuning, 10 validation. Six HUMOTO clips have historical V2 exposure; two HUMOTO and all six HiPHI are new to this V2 study. Validation means unused for this weight choice, not globally unseen.',
        current='Geometry side, regional gap, signed recovery OFF, point attraction OFF, MuJoCo collision distance, collision-aware frame-zero initializer, subsequent motion compensation; body collision ON',
        candidates={'current':{},'tip_relaxed':{'fingertip_weight':6.25},'surface_emphasis':{'gap_weight':1024.}},
        candidate_scope='One shared weight change at a time; no per-object/per-sequence weights. All other settings, targets, masks, collisions and initialization algorithm fixed.',
        selection='Equal-clip mean intended original-mesh patch gap on tuning clips; reject if fewer solves, mean peak penetration rises >1 mm, or mean arm jerk rises >25%. If none qualifies retain current. Lock selection before validation candidate runs.',
        colliders='Validated component cart/stool reused. Other objects use isolated UMR/GRAIL-parameter CoACD with maximum32 hulls, or one convex hull only if already convex. Geometry error audited and reported, not presumed valid.',
        scaling='Human, object mesh and world trajectory uniformly scaled by robot/actor height about the existing source floor. No independent toe/object regrounding.',
        evaluation='Existing frozen V2 metrics and thresholds unchanged; HiPHI masks derived from its actual fitted landmarks/skin. Dataset comparisons are not paired source-representation ablations.',
        resources=dict(gpu=False,workers=2,cpu_quota_percent=200,memory_max_gib=5,nice=15),
        limitations=['Pilot windows, not full clips or whole datasets.','Kinematic contact does not establish force closure or physical grasp success.','Proxy surface error can bias collision contact; original-mesh patch gaps reported separately.']))
    write(OUT/'status.json',dict(state='prepared',sequences=len(result),completed=0))
if __name__=='__main__':main()
