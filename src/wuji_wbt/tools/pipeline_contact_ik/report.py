"""Local review page for the organizer pilot; no generated results go into Git."""
import csv
import html
from config import *


def main():
    results=[]
    for variant in REMOVALS:
        dest=OUT/'variants'/variant/'clips'/SEQ
        result={'variant':variant,**read(dest/'metrics.json')['v2'],**read(dest/'ik_diagnostics.json')}
        if (dest/'physical_contact_audit.json').exists():result.update(read(dest/'physical_contact_audit.json'))
        results.append(result)
    rows=[('contact_f1_pct','Contact F1 at 30 mm (%) ↑'),('contact_recall_pct','Contact recall (%) ↑'),
        ('contact_f1_penetration_gated_pct','Contact F1 with 1 mm depth gate (%) ↑'),
        ('tip_surface_recall_2mm_pct','Supplement: tip collision-surface contact recall at 2 mm (%) ↑'),
        ('tip_surface_f1_2mm_pct','Supplement: tip collision-surface contact F1 at 2 mm (%) ↑'),
        ('contact_location_gap_mm','Source contact-point gap (mm) ↓'),
        ('fingertip_surface_distance_on_contact_mm','Fingertip surface gap (mm) ↓'),
        ('combined_pen_duration_pct','Frozen combined penetration duration (%) ↓'),
        ('combined_pen_conditional_depth_cm','Frozen conditional penetration depth (cm) ↓'),
        ('left_torso_min_clearance_mm','Supplement: left torso minimum clearance (mm) ↑'),
        ('right_torso_min_clearance_mm','Supplement: right torso minimum clearance (mm) ↑'),
        ('left_torso_contact_mean_clearance_mm','Supplement: mean left arm/hand-to-torso clearance during contact (mm) ↑'),
        ('torso_clearance_below5mm_hand_frames_pct','Supplement: contact hand-frames below 5 mm torso clearance (%) ↓'),
        ('collision_feasible_frames','Supplement: controllable collision-feasible frames / 227'),
        ('frozen_collision_frames','Supplement: frames with frozen geometry violations'),
        ('object_max_depth_mm','Supplement: Coal maximum object depth (mm) ↓'),
        ('self_max_depth_mm','Supplement: Coal maximum self depth incl. frozen fingers (mm) ↓'),
        ('arm_min_margin_contact_mean_deg','Mean minimum arm joint margin during contact (°) ↑'),
        ('arm_contact_near_limit5deg_pct','Contact hand-frames near arm limit (%) ↓'),
        ('arm_contact_acceleration_rms','Arm acceleration RMS during contact (rad/s²) ↓'),
        ('arm_contact_jerk_rms','Arm jerk RMS during contact (rad/s³) ↓'),
        ('fixed_finger_rigid_fit_lower_bound_rms_mm','Optimistic fixed-finger contact RMS lower bound (mm)')]
    def fmt(v):return 'N/A' if v is None else f'{v:.3f}'
    table='<table><tr><th>Metric</th><th>Saved V2</th><th>Contact-first arm IK</th></tr>'
    for key,label in rows:table+='<tr><td>'+html.escape(label)+'</td>'+''.join('<td>'+fmt(r.get(key))+'</td>' for r in results)+'</tr>'
    table+='</table>'
    with (OUT/'comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['metric','baseline','contact_ik'])
        for k,label in rows:w.writerow([k,*[r.get(k) for r in results]])
    run=read(OUT/'variants/contact_ik/clips'/SEQ/'run.json')
    summary=dict(sequence=SEQ,frames=run['frames'],solve_seconds=run['seconds'],
        metrics={r['variant']:{k:r.get(k) for k,_ in rows} for r in results},
        frozen_body_and_fingers_exact=all(r['frozen_coordinates_exact'] for r in results),
        feasible_frames=run['feasible_frames'],complete_collision_free_motion=False if run['frozen_geometry_violation_frames'] else run['feasible_frames']==run['frames'],
        scope='One development clip. Geometric experiment, not a replicated DexWeave implementation or policy test.')
    write(OUT/'summary.json',summary)
    page='''<!doctype html><meta charset="utf-8"><title>Organizer contact-first arm IK</title>
<style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;color:#173246}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #ccd5dd}th{background:#edf3f8}td:not(:first-child){text-align:right}video,img{max-width:100%}.note{background:#fff2d8;padding:15px}</style>
<h1>Organizer: fixed-finger contact-first arm IK</h1>
<p>Human demonstration | saved V2 | contact-first IK. Both robot trajectories share exactly the same per-frame finger angles, root, waist, legs and object. Only 14 arm/wrist coordinates may change. No interaction mesh or reference-arm-angle objective is used.</p>
<p>Source contact anchors and masks are unchanged. The solver first recovers collision feasibility, then minimizes contact-point error, then uses the primary task nullspace for joint-range posture, weak palm orientation and temporal correction preferences. Torso clearance is 5 mm; other registered collisions have a zero margin.</p>
<p class="note">The frozen comparison metrics retain their original MuJoCo backend and definitions, including the previously diagnosed native distance artifact. Separately labelled clearance diagnostics use independent Coal convex queries on both trajectories. Frozen finger geometry can contain collisions that arm-only IK cannot repair. A completed trajectory is not automatically a feasible or improved result.</p>'''
    page+=f'<p>Full clip: {run["frames"]} frames at 30 Hz. IK process: {run["seconds"]:.1f} s. Controllable collision-feasible frames: {run["feasible_frames"]}/{run["frames"]}. Frozen-geometry violation frames: {run["frozen_geometry_violation_frames"]}.</p>'+table
    page+='<p class="note">This fixed-finger test changes collision enforcement and removes the mesh, so it does not isolate a single objective term. A first attempt using MuJoCo libccd is archived separately because independent convex tests disproved some reported collisions; the final pilot uses Coal without changing targets or objective weights.</p>'
    page+='<p>Supplemental tip-surface contact uses actual fingertip collision convexes and CoACD object pieces: signed gap ≤2 mm, penetration ≤10 micrometres, evaluated against the same source per-tip mask. This differs from the frozen landmark-distance contact metric. Both definitions remain visible.</p>'
    page+=f'<h2>Same-time comparison</h2><video controls preload="metadata" src="renders/{SEQ}.mp4" poster="renders/{SEQ}.png"></video>'
    page+=f'<p><a href="renders/{SEQ}.mp4">Three-panel MP4</a> · <a href="renders/{SEQ}_posture.png">Arm posture close-up</a> · <a href="renders/{SEQ}_hands.png">Hand close-up</a></p>'
    page+=f'<img src="renders/{SEQ}_posture.png"><p><a href="comparison.csv">Metrics CSV</a> · <a href="summary.json">Summary</a> · <a href="protocol.json">Protocol and input hashes</a> · <a href="diagnostic_summary.json">Detailed feasibility diagnostics</a></p>'
    page+='<p>The rigid-fit lower bound allows each frozen hand to translate/rotate freely, ignoring arm reach and collisions. Nonzero error means the fixed fingertip constellation cannot exactly match every source anchor even with unlimited arm freedom. The targets track fingertip landmark centers; physical surface contact is measured separately.</p>'
    if (OUT/'physical_contact_audit.json').exists():page+='<p><a href="physical_contact_audit.json">Independent convex-intersection checks and physical contact audit</a></p>'
    (OUT/'index.html').write_text(page)


if __name__=='__main__':main()
