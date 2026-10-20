"""Matched frozen metrics, explicitly separate event diagnostics, local video index."""
import csv
import html
import numpy as np
from config import *

def main():
    from solver import load
    rows=[]
    for seq in COHORT:
        _,_,_,ref,_,_=load(seq,'baseline')
        fingers=np.concatenate(ref.finger_columns);arms=np.concatenate(ref.arm_columns)
        lo,hi=(105,120) if 'organizer' in seq else (73,90)
        for variant in VARIANTS:
            dest=destination(seq,variant)
            if not (dest/'geometry_diagnostics.json').exists():continue
            d=read(dest/'diagnostics.json');g=read(dest/'geometry_diagnostics.json');r=read(dest/'run.json')
            Q=np.load(dest/'final_v2.npz')['qpos'][:,ref.qadr]
            raw=np.load(dest/'contact_geometry_per_frame.npz');depth=raw['max_depth_object_ground_self_m']*1000
            velocity=np.vstack([np.zeros((1,ref.n)),np.diff(Q,axis=0)])*30
            acceleration=np.vstack([np.zeros((2,ref.n)),np.diff(Q,n=2,axis=0)])*900
            jerk=np.vstack([np.zeros((3,ref.n)),np.diff(Q,n=3,axis=0)])*27000
            row=dict(sequence=seq,variant=variant,frames=d['frames'],physical_f1_pct=d['physical_touch_vs_original_source']['f1_pct'],
                strict_f1_pct=d['strict_physical_touch_vs_original_source']['f1_pct'],physical_recall_pct=d['physical_touch_vs_original_source']['recall_pct'],
                intended_patch_recall_pct=d['intended_patch_touch_recall_skin_contact_pct'],patch_gap_mm=d['physical_patch_gap_mean_mm'],
                geometry_correct_side_pct=g['geometry_correct_side_pct'],geometry_intended_contact_pct=g['geometry_intended_contact_pct'],
                contact_tip_mean_pen_mm=g['contact_tip_mean_pen_mm'],contact_tip_max_pen_mm=g['contact_tip_max_pen_mm'],
                object_mean_frame_max_pen_mm=d['object_mean_frame_max_penetration_mm'],object_max_pen_mm=d['object_max_penetration_mm'],
                object_above1mm_frame_pct=d['object_penetration_above_1mm_pct'],object_above5mm_frame_pct=d['object_penetration_above_5mm_pct'],
                self_max_pen_mm=d['self_max_penetration_mm'],ground_max_pen_mm=d['ground_max_penetration_mm'],
                arm_accel_rad_s2=d['arm_acceleration_rms_rad_s2'],arm_jerk_rad_s3=d['arm_jerk_rms_rad_s3'],
                finger_accel_rad_s2=d['finger_acceleration_rms_rad_s2'],finger_jerk_rad_s3=float(np.sqrt(np.mean(jerk[3:,fingers]**2))),
                arm_near_limit_pct=d['arm_contact_near_limit5deg_pct'],drift_cm_s=d['relative_surface_witness_drift_cm_s'],
                event_first_frame=lo,event_last_frame=hi,event_object_max_pen_mm=float(depth[lo:hi+1,0].max()),
                event_finger_max_change_deg=float(np.degrees(np.abs(velocity[lo:hi+1,fingers]).max()/30)),
                event_finger_accel_max_rad_s2=float(np.abs(acceleration[lo:hi+1,fingers]).max()),
                event_finger_jerk_rms_rad_s3=float(np.sqrt(np.mean(jerk[lo:hi+1,fingers]**2))),
                event_arm_jerk_rms_rad_s3=float(np.sqrt(np.mean(jerk[lo:hi+1,arms]**2))),
                solve_seconds=r['joint_refinement_seconds'],qp_solver_seconds=r['solver_seconds'],
                accepted_substeps=r['accepted_substeps'],guard_unresolved_substeps=r['guard_unresolved_substeps'],
                resource_cap_fallback_frames=r['resource_cap_fallback_frames'])
            rows.append(row)
    if not rows:return
    with (OUT/'per_sequence.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(OUT/'summary.json',dict(rows=rows,completed=len(rows),total=8,
        definitions='Frozen historical and contact-geometry metrics unchanged. F1 uses <=2 mm separation and <=1 mm penetration. Strict F1 uses <=0.01 mm penetration. Angular finite differences at30Hz. Supplemental event windows organizer105..120 inclusive and rack73..90 inclusive are fixed from prior diagnosis. Runtime includes predictor and continuation; body and finger preprocessing are reused.',
        caveats='Only two development clips. Substeps have more QP iterations available and bounded adaptive rejection with explicitly reported unresolved guards. No object trajectory optimization.'))
    improvements=[]
    for seq in COHORT:
        controls=[r for r in rows if r['sequence']==seq and r['variant']=='baseline']
        if not controls:continue
        control=controls[0]
        for row in rows:
            if row['sequence']!=seq or row['variant']=='baseline':continue
            result=dict(sequence=seq,variant=row['variant'])
            for key in ('object_max_pen_mm','event_object_max_pen_mm','event_finger_jerk_rms_rad_s3','finger_accel_rad_s2','arm_accel_rad_s2','solve_seconds'):
                result[key+'_change_pct']=100*(row[key]/control[key]-1) if control[key] else None
            result['physical_f1_change_percentage_points']=row['physical_f1_pct']-control['physical_f1_pct']
            improvements.append(result)
    write(OUT/'changes_vs_baseline.json',improvements)
    columns=['variant','physical_f1_pct','strict_f1_pct','patch_gap_mm','object_max_pen_mm','contact_tip_mean_pen_mm',
             'event_object_max_pen_mm','event_finger_max_change_deg','finger_accel_rad_s2','arm_accel_rad_s2','solve_seconds',
             'guard_unresolved_substeps']
    doc=['<!doctype html><meta charset="utf-8"><title>Geometry side: motion initialization and continuation</title>',
         '<style>body{font:16px system-ui;margin:25px;max-width:2400px}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #bbb;padding:8px}video,img{max-width:100%}.scroll{overflow:auto}</style>',
         '<h1>Human / geometry baseline / motion init / collision substeps / combined</h1>',
         '<p>Geometry side + regional gap, signed recovery OFF, thumb/index x1 in all conditions. Existing signed collision checking remains ON. Same scaled object and fixed trajectory, source targets, objectives and evaluation definitions. Internal continuation has extra solve work and collision acceptance checks; it does not optimize the object trajectory.</p>',
         '<p>Both clips have geometry scale 0.8188585607940446. Position transform is s*(position - [0,0,z_min]); rotations unchanged. Body/object outputs must match the inherited Omni pass exactly.</p>',
         '<p><a href="protocol.json">Protocol</a> · <a href="status.json">Status</a> · <a href="per_sequence.csv">All metrics</a> · <a href="changes_vs_baseline.json">Changes</a> · <a href="validation.json">Validation</a> · <a href="contact_threshold_sensitivity.csv">Threshold sensitivity</a></p>',
         '<p>Contact F1 allows 2 mm separation and 1 mm penetration; strict F1 allows only 0.01 mm penetration. Event maxima are explicitly supplementary. Full-clip maxima are also shown so improvements cannot hide a shifted glitch. Guard-unresolved counts identify intervals accepted at the minimum step or after the resource cap, rather than claiming strict continuous collision feasibility.</p>']
    for seq in COHORT:
        doc.append('<h2>'+html.escape(seq)+'</h2><div class="scroll"><table><tr>'+''.join('<th>'+k+'</th>' for k in columns)+'</tr>')
        for row in rows:
            if row['sequence']!=seq:continue
            doc.append('<tr>'+''.join('<td>'+html.escape(f'{row[k]:.3f}' if isinstance(row[k],float) else str(row[k]))+'</td>' for k in columns)+'</tr>')
        doc.append('</table></div>')
        for suffix,label in (('','Whole body'),('_hands','Hands'),('_thumb','Right thumb')):
            path=OUT/'renders'/f'{seq}{suffix}.mp4'
            if path.exists():doc.append(f'<p>{label}: <a href="renders/{path.name}">MP4</a></p><video controls preload="metadata" src="renders/{path.name}"></video>')
    (OUT/'index.html').write_text('\n'.join(doc))

if __name__=='__main__':main()
