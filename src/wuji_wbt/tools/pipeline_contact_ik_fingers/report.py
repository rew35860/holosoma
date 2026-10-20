"""Local three-method comparison; generated reports and media stay outside Git."""
import csv
import html
from config import *


def main():
    results={}
    for variant in REMOVALS:
        dest=OUT/'variants'/variant/'clips'/SEQ
        results[variant]={**read(dest/'metrics.json')['v2'],**read(dest/'ik_diagnostics.json'),
                          **read(dest/'physical_contact_audit.json')}
    rows=[
        ('contact_f1_pct','Frozen contact F1 at 30 mm (%) ↑'),
        ('contact_recall_pct','Frozen contact recall at 30 mm (%) ↑'),
        ('contact_f1_penetration_gated_pct','Frozen F1 with 1 mm penetration gate (%) ↑'),
        ('tip_surface_recall_2mm_pct','Coal tip-surface contact recall at 2 mm (%) ↑'),
        ('tip_surface_f1_2mm_pct','Coal tip-surface contact F1 at 2 mm (%) ↑'),
        ('contact_location_gap_mm','Source contact-point gap (mm) ↓'),
        ('fingertip_surface_distance_on_contact_mm','Fingertip landmark surface gap (mm) ↓'),
        ('primary_tip_error_mm','Thumb/index source fingertip alignment (mm)'),
        ('secondary_tip_error_mm','Other-finger source fingertip alignment (mm)'),
        ('palm_normal_error_deg','Source palm-normal difference (°)'),
        ('relative_hand_orientation_increment_error_deg','Object-relative hand rotation increment difference (°)'),
        ('combined_pen_duration_pct','Frozen combined penetration duration (%) ↓'),
        ('combined_pen_conditional_depth_cm','Frozen conditional penetration depth (cm) ↓'),
        ('object_max_depth_mm','Coal registered object maximum depth (mm) ↓'),
        ('self_max_depth_mm','Coal registered self maximum depth (mm) ↓'),
        ('all_registered_clearances_feasible_frames','All registered clearance requirements satisfied (frames / 227) ↑'),
        ('left_torso_contact_mean_clearance_mm','Mean left arm/hand–torso clearance during contact (mm) ↑'),
        ('left_torso_min_clearance_mm','Minimum left arm/hand–torso clearance (mm) ↑'),
        ('right_torso_min_clearance_mm','Minimum right arm/hand–torso clearance (mm) ↑'),
        ('torso_clearance_below5mm_hand_frames_pct','Contact hand-frames below 5 mm torso clearance (%) ↓'),
        ('arm_min_margin_contact_mean_deg','Mean minimum arm joint margin during contact (°) ↑'),
        ('arm_contact_near_limit5deg_pct','Contact hand-frames within 5° of arm limit (%) ↓'),
        ('arm_contact_acceleration_rms','Arm contact acceleration RMS (rad/s²) ↓'),
        ('arm_contact_jerk_rms','Arm contact jerk RMS (rad/s³) ↓'),
        ('finger_change_rms_deg','Finger change from saved V2 RMS (°)'),
        ('finger_change_max_deg','Maximum finger change from saved V2 (°)'),
        ('finger_acceleration_rms','Finger acceleration RMS, whole clip (rad/s²) ↓'),
    ]
    for kind in ('object','self'):
        for mm in (.01,.1,1,5,10,20):
            rows.append((f'coal_{kind}_duration_above_{mm:g}mm_pct',f'Coal registered {kind} penetration above {mm:g} mm (% frames) ↓'))
    table='<table><tr><th>Metric</th>'+''.join('<th>'+html.escape(v)+'</th>' for v in LABELS.values())+'</tr>'
    for key,label in rows:
        values=[results[v].get(key) for v in REMOVALS]
        table+='<tr><td>'+html.escape(label)+'</td>'+''.join('<td>'+('N/A' if x is None else f'{x:.3f}')+'</td>' for x in values)+'</tr>'
    table+='</table>'
    with (OUT/'comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['metric',*REMOVALS])
        writer.writerows([key,*[results[v].get(key) for v in REMOVALS]] for key,_ in rows)
    run=read(OUT/'variants/contact_ik/clips'/SEQ/'run.json')
    old=read(FIXED/'variants/contact_ik/clips'/SEQ/'run.json')
    write(OUT/'summary.json',dict(sequence=SEQ,frames=run['frames'],
        solve_seconds=dict(fixed_fingers=old['seconds'],adjustable_fingers=run['seconds']),
        frozen_body_object_exact=all(v['frozen_coordinates_exact'] for v in results.values()),
        metrics={v:{k:r.get(k) for k,_ in rows} for v,r in results.items()},
        feasibility_caution='Registered arm/hand pairs only; solver completion is not a proof of collision freedom or globally optimal posture. One development clip.'))
    page='''<!doctype html><meta charset="utf-8"><title>Organizer: contact-first IK with adjustable fingers</title>
<style>body{font:16px system-ui;max-width:1920px;margin:25px auto;padding:20px;color:#173246}table{border-collapse:collapse}td,th{border:1px solid #ccd5dd;padding:8px}td:not(:first-child){text-align:right}video,img{max-width:100%}.note{background:#fff2d8;padding:16px}</style>
<h1>Organizer: allow finger adjustment</h1>
<p>Human demonstration | saved V2 | contact-first IK with fingers fixed | contact-first IK with arms and fingers adjustable. Root, waist, legs and object are identical in every robot panel.</p>
<p>The new solve uses the same source contact points, masks, collision geometry and margins, hierarchy, iteration limits and weights as the previous arm-only pilot. It adds the 40 finger coordinates to the 14 arm/wrist coordinates. The existing temporal correction preference, damping and bounds extend to these new variables. The joint-center posture preference still acts only on the arms. No interaction mesh, Omni arm prior, bone term or finger initialization prior is added.</p>
<p class="note">Frozen metrics retain the original evaluator and its known MuJoCo distance artifacts. Separately labelled Coal metrics use identical convex geometry and registered pair filters for all three results. All-pair feasibility includes zero-gradient collisions, making that count comparable despite different optimizer freedoms. It is not a full-body collision certificate.</p>'''
    before=results['fixed_fingers'];after=results['contact_ik']
    page+=f'<p class="note"><strong>Mixed result:</strong> compared with fixed fingers, strict surface-contact F1 improves from {before["tip_surface_f1_2mm_pct"]:.1f}% to {after["tip_surface_f1_2mm_pct"]:.1f}%, and maximum registered self-penetration falls from {before["self_max_depth_mm"]:.2f} to {after["self_max_depth_mm"]:.2f} mm. However, object penetration above 1 mm occurs in {after["coal_object_duration_above_1mm_pct"]:.1f}% of frames versus {before["coal_object_duration_above_1mm_pct"]:.1f}% previously. {run["frames"]-after["all_registered_clearances_feasible_frames"]} frames still miss a clearance requirement. The hands rotate and change shape substantially; inspect the video before treating this as a natural grasp. This is a diagnostic candidate, not an automatically accepted replacement.</p>'
    page+=f'<p>Full clip: {run["frames"]} frames at 30 Hz. Fixed-finger solve: {old["seconds"]:.1f} s; adjustable-finger solve: {run["seconds"]:.1f} s.</p>'+table
    page+='<p>Coal tip-surface contact requires signed gap ≤2 mm and penetration ≤10 micrometres, scored against the original per-tip source contact mask. The source contact-point gap instead measures landmark-to-specific-anchor distance. These are distinct quantities.</p>'
    page+='<p>Fingertip alignment uses world-frame robot fingertips against source human fingertips on source-contact hand-frames; palm alignment uses the same hand-frame mask. These describe imitation of the source, not whether a different robot posture is comfortable or contact is physically stable. No grasp-force or dynamic stability test is performed.</p>'
    page+=f'<h2>Synchronized comparison</h2><video controls preload="metadata" src="renders/{SEQ}.mp4" poster="renders/{SEQ}.png"></video>'
    page+=f'<p><a href="renders/{SEQ}.mp4">Four-panel MP4</a> · <a href="renders/{SEQ}_hands.png">Hand close-up</a> · <a href="renders/{SEQ}_posture.png">Arm posture close-up</a></p>'
    page+=f'<img src="renders/{SEQ}_hands.png">'
    page+='<p><a href="comparison.csv">Metrics CSV</a> · <a href="summary.json">Summary</a> · <a href="protocol.json">Protocol</a> · <a href="diagnostic_summary.json">Detailed diagnostics</a> · <a href="physical_contact_audit.json">Physical contact audit</a></p>'
    (OUT/'index.html').write_text(page)


if __name__=='__main__':main()
