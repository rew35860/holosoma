"""Local four-panel comparison and unchanged metrics with explicit supplements."""
import csv
import html
from config import *


def main():
    rows=[]
    for seq in COHORT:
        for variant in VARIANTS:
            dest=destination(seq,variant)
            if not (dest/'geometry_diagnostics.json').exists():continue
            d=read(dest/'diagnostics.json');g=read(dest/'geometry_diagnostics.json');r=read(dest/'run.json')
            rows.append(dict(sequence=seq,variant=variant,frames=d['frames'],
                physical_f1_pct=d['physical_touch_vs_original_source']['f1_pct'],
                strict_f1_pct=d['strict_physical_touch_vs_original_source']['f1_pct'],
                physical_recall_pct=d['physical_touch_vs_original_source']['recall_pct'],
                intended_patch_recall_pct=d['intended_patch_touch_recall_skin_contact_pct'],
                patch_gap_mm=d['physical_patch_gap_mean_mm'],gap_target_mae_mm=d['surface_gap_target_mae_mm'],
                reference_correct_side_pct=d['correct_side_contact_pct'],
                geometry_correct_side_pct=g['geometry_correct_side_pct'],
                geometry_intended_contact_pct=g['geometry_intended_contact_pct'],
                contact_tip_mean_pen_mm=g['contact_tip_mean_pen_mm'],contact_tip_max_pen_mm=g['contact_tip_max_pen_mm'],
                contact_tip_above1mm_pct=g['contact_tip_pen_above_1mm_pct'],
                object_mean_frame_max_pen_mm=d['object_mean_frame_max_penetration_mm'],
                object_max_pen_mm=d['object_max_penetration_mm'],object_above1mm_frame_pct=d['object_penetration_above_1mm_pct'],
                self_mean_frame_max_pen_mm=d['self_mean_frame_max_penetration_mm'],self_max_pen_mm=d['self_max_penetration_mm'],
                ground_max_pen_mm=d['ground_max_penetration_mm'],
                arm_accel_rad_s2=d['arm_acceleration_rms_rad_s2'],arm_jerk_rad_s3=d['arm_jerk_rms_rad_s3'],
                finger_accel_rad_s2=d['finger_acceleration_rms_rad_s2'],arm_near_limit_pct=d['arm_contact_near_limit5deg_pct'],
                drift_cm_s=d['relative_surface_witness_drift_cm_s'],solve_seconds=r['joint_refinement_seconds'],
                qp_solver_seconds=r['solver_seconds'],line_search_stopped_frames=r['line_search_stopped_frames']))
    if rows:
        with (OUT/'per_sequence.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(OUT/'summary.json',dict(rows=rows,completed=len(rows),total=6,
        interpretation='Two development clips, three newly solved variants. Signed recovery absent throughout; signed collision constraints unchanged. Primary metric definitions unchanged; geometry-side/contact and tip-only depths are explicit supplements. Runtime is refinement only, on a shared CPU-limited workstation.'))
    doc=['<!doctype html><meta charset="utf-8"><title>V2 geometry side without signed recovery</title>',
        '<style>body{font:16px system-ui;margin:30px;max-width:2000px}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #bbb}video,img{max-width:100%}.table{overflow:auto}</style>',
        '<h1>Human / current V2 / geometry side / geometry side + thumb-index ×4</h1>',
        '<p>Signed-recovery objective is OFF in every robot panel. Signed Coal collision checking, 1 mm tolerance and collision slack remain ON. All body/object trajectories, human fingertip tracking, source targets, initialization and solver settings are identical.</p>',
        '<p>Baseline uses the original reference-point side preference plus regional surface gap. Geometry side replaces that preference with min(0, min_vertex n·(v−anchor) + 0.001). Priority multiplies only side and surface-gap squared costs by four for thumb and index on BOTH hands during source contact. Other digits retain their weights; no hard thumb constraint is added.</p>',
        '<p>Human fingertip tracking stays 25; interaction mesh stays 500. Side/gap weights are 2500/256, or 10000/1024 for primary digits in the priority variant. Source skin-gap targets are unchanged, including positive gaps.</p>',
        '<p><a href="protocol.json">Protocol</a> · <a href="status.json">Status</a> · <a href="per_sequence.csv">All metrics CSV</a> · <a href="validation.json">Validation</a> · <a href="contact_threshold_sensitivity.csv">Threshold sensitivity</a></p>',
        '<p>Contact F1 permits ≤2 mm separation and ≤1 mm penetration. Strict F1 permits ≤0.01 mm penetration. Original intended-patch recall uses source-skin-contact frames and the reference-point side test. Supplemental geometry contact uses the whole-tip side test and all original source-contact fingertip-frames. These denominators differ; neither score proves a physical grasp.</p>']
    columns=['variant','physical_f1_pct','strict_f1_pct','intended_patch_recall_pct','geometry_correct_side_pct',
             'geometry_intended_contact_pct','contact_tip_mean_pen_mm','contact_tip_max_pen_mm',
             'object_max_pen_mm','self_max_pen_mm','patch_gap_mm','arm_accel_rad_s2','solve_seconds']
    for seq in COHORT:
        doc.append('<h2>'+html.escape(seq)+'</h2><div class="table"><table><tr>'+''.join('<th>'+k+'</th>' for k in columns)+'</tr>')
        for row in [r for r in rows if r['sequence']==seq]:
            doc.append('<tr>'+''.join('<td>'+html.escape(f'{row[k]:.3f}' if isinstance(row[k],float) else str(row[k]))+'</td>' for k in columns)+'</tr>')
        doc.append('</table></div>')
        for suffix,label in (('', 'Whole-body comparison'),('_hands','Hand close-up'),('_thumb','Right-thumb focus')):
            path=OUT/'renders'/f'{seq}{suffix}.mp4'
            if path.exists():doc.append(f'<p>{label}: <a href="renders/{path.name}">Download MP4</a></p><video controls preload="metadata" src="renders/{path.name}"></video>')
        for suffix in ('','_hands','_thumb','_posture'):
            path=OUT/'renders'/f'{seq}{suffix}.png'
            if path.exists():doc.append(f'<p><img src="renders/{path.name}"></p>')
    (OUT/'index.html').write_text('\n'.join(doc))


if __name__=='__main__':main()
