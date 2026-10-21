"""Equal-clip summaries with explicit failure denominators and fixed definitions."""
import csv,html
import numpy as np
from config import *

def main():
    records=[];failures=[]
    selection=read(OUT/'selection.json') if (OUT/'selection.json').exists() else {'selected':'pending'}
    for r in rows():
        for v in VARIANTS:
            folder=OUT/'variants'/v/'clips'/r['sequence']
            if not (folder/'evaluate_completed.json').exists():continue
            d=read(folder/'diagnostics.json');g=read(folder/'geometry_diagnostics.json');m=read(folder/'metrics.json')['v2'];run=read(folder/'run.json')
            whole=read(folder/'whole_body_collision.json')
            source_contact=np.load(OUT/'evaluation_v1/per_frame'/f"{r['sequence']}.npz")['source_contact']
            entry=dict(sequence=r['sequence'],dataset=r['dataset'],object=r['object'],split=r['split'],variant=v,frames=run['frames'],
                source_contact_frames=int(source_contact.any(axis=(1,2)).sum()),
                source_contact_frame_pct=100.*float(source_contact.any(axis=(1,2)).mean()),
                patch_gap_mm=d['physical_patch_gap_mean_mm'],gap_target_mae_mm=d['surface_gap_target_mae_mm'],
                physical_f1_pct=d['physical_touch_vs_original_source']['f1_pct'],physical_recall_pct=d['physical_touch_vs_original_source']['recall_pct'],
                strict_physical_f1_pct=d['strict_physical_touch_vs_original_source']['f1_pct'],intended_contact_pct=g['geometry_intended_contact_pct'],
                hand_object_peak_mm=d['object_max_penetration_mm'],hand_object_mean_frame_max_mm=d['object_mean_frame_max_penetration_mm'],
                hand_object_duration_gt1mm_pct=d['object_penetration_above_1mm_pct'],self_peak_mm=d['self_max_penetration_mm'],
                whole_object_peak_mm=whole['all_robot_object']['peak_mm'],leg_object_peak_mm=whole['legs_object']['peak_mm'],
                ground_peak_mm=whole['all_robot_ground']['peak_mm'],combined_pen_duration_pct=m['combined_pen_duration_pct'],
                conditional_depth_cm=m['combined_pen_conditional_depth_cm'],primary_tip_error_mm=m['primary_tip_error_mm'],
                palm_normal_deg=m['palm_normal_error_deg'],foot_skating_cm_s=m['foot_skating_mean_cm_s'],
                arm_jerk_rad_s3=d['arm_jerk_rms_rad_s3'],finger_accel_rad_s2=d['finger_acceleration_rms_rad_s2'],
                near_arm_limit_pct=d['arm_contact_near_limit5deg_pct'],refinement_seconds=run['joint_refinement_seconds'],
                initialization_seconds=run['initializer_seconds'],body_seconds=read(marker(r['sequence'],'body'))['seconds'])
            records.append(entry)
    for f in (OUT/'attempts').glob('*.json'):
        x=read(f)
        if x['returncode']:
            log=OUT/'logs'/x['sequence']/f"{x['variant']}_{x['stage']}.log"
            x['error_tail']=log.read_text()[-2500:];failures.append(x)
    summaries=[]
    for dataset in ('humoto','hiphi'):
        for split in ('tune','validation'):
            for v in VARIANTS:
                subset=[r for r in records if r['dataset']==dataset and r['split']==split and r['variant']==v]
                if not subset:continue
                item=dict(dataset=dataset,split=split,variant=v,completed=len(subset),planned=sum(r['dataset']==dataset and r['split']==split for r in rows()))
                for k in subset[0]:
                    vals=[r[k] for r in subset if isinstance(r[k],(float,int))]
                    if vals:item[k]=float(np.mean(vals))
                summaries.append(item)
    for name,data in [('per_sequence',records),('summary',summaries)]:
        write(OUT/f'{name}.json',data)
        if data:
            with (OUT/f'{name}.csv').open('w') as stream:
                w=csv.DictWriter(stream,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write(OUT/'failures.json',failures)
    recovery=[]
    for f in (OUT/'recovery_audit').glob('*/original_body_attempt.json'):
        seq=f.parent.name
        recovery.append(dict(sequence=seq,original_attempt=read(f),
            now_evaluated=(OUT/'variants/current/clips'/seq/'evaluate_completed.json').exists(),
            retry_log=str(OUT/'clips'/seq/'body_retries.json')))
    write(OUT/'recovery_summary.json',recovery)
    geometry=[]
    for f in sorted((OUT/'geometry_audit').glob('*.json')):
        x=read(f);build=read(OUT/'geometry'/x['object']/'build.json')
        geometry.append(dict(sequence=x['sequence'],object=x['object'],pieces=x['pieces'],
            method=build['method'],preparation_seconds=build['seconds'],
            timing_scope='Load/copy cached hulls' if build['method'].startswith('Previously') else 'Fresh collider construction',
            median_error_mm=x['anchor_absolute_error_mm']['median'],p95_error_mm=x['anchor_absolute_error_mm']['p95'],
            maximum_error_mm=x['anchor_absolute_error_mm']['maximum'],sampled_half_mm_pass=x['passes_sampled_half_mm_budget']))
    write(OUT/'collider_summary.json',geometry)
    def table(data,columns):
        return '<table><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{r.get(c):.2f}' if isinstance(r.get(c),float) else str(r.get(c,'')))+'</td>' for c in columns)+'</tr>' for r in data)+'</table>'
    doc='''<!doctype html><meta charset="utf-8"><title>V2 cross-dataset pilot</title><style>body{font:16px system-ui;max-width:1500px;margin:auto;padding:30px}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #bbb}video{width:100%}section{margin:30px 0}pre{white-space:pre-wrap}</style><h1>Current V2 across HUMOTO and HiPHI</h1><p>14 preselected windows of up to eight seconds, 30 FPS: eight HUMOTO and six HiPHI. Four clips tune one shared weight; the other ten validate that weight. Native HUMOTO skin; HiPHI articulated SMPL-X pose-plus-shape fit. Uniform world scaling and source floor. CPU-only, active object colliders in the body pass. This pilot does not establish physical grasp success or whole-dataset generalization. Some clips were used in earlier V2 experiments; validation means unused for this weight choice, not globally unseen.</p>'''
    doc+='<p>Selected shared setting: <b>'+html.escape(selection['selected'])+'</b>. <a href="protocol.json">Protocol and definitions</a>; <a href="per_sequence.csv">per-clip CSV</a>; <a href="failures.json">all failures</a>.</p>'
    if (OUT/'recommendation.json').exists():
        rec=read(OUT/'recommendation.json')
        doc+='<p>Provisional shared preset after validation: <b>'+html.escape(rec['recommended'])+'</b>. <a href="recommendation.json">Validation checks</a>; <a href="recommended_preset.json">shared preset</a>. Only the regional surface-gap weight changes from 256 to 1024; the selected preset retains the current initializer, geometry-side term, fingertip tracking and collision settings.</p>'
        paired=[]
        for group in rec['per_dataset']:
            for label,values in group['means'].items():
                paired.append(dict(dataset=group['dataset'],setting=label,paired_clips=group['paired'],**values))
        doc+='<h2>Weight-validation comparison</h2>'+table(paired,['dataset','setting','paired_clips','patch_gap_mm','physical_f1_pct','intended_contact_pct','hand_object_peak_mm','self_peak_mm','arm_jerk_rad_s3'])
        doc+='<p>These are equal-clip averages. Average improvements do not imply every clip improves: HUMOTO working-chair and shelf physical F1 decrease. The first eight seconds of HiPHI clothes-rack contain only two source-contact frames; its zero F1 and worse gap do not establish sustained grasp quality. A longer contact window is needed. HiPHI table and chair have 50 and 54 source-contact frames. Strict intended-contact coverage remains low on several objects. The preset is a useful pilot result, not a grasp guarantee. Organizer solver recovery and clothes-rack source-mesh repair were informed by failures in this cohort; they are not independently validated generalization results.</p>'
    if (OUT/'validation.json').exists():
        doc+='<p><a href="validation.json">Independent artifact checks</a> and <a href="provenance.json">code/input hashes and versions</a>: finite motions, joint limits, exact unchanged object trajectories, consistent source-target hashes and expected variant weights.</p>'
    doc+=table(summaries,['dataset','split','variant','completed','planned','patch_gap_mm','physical_f1_pct','intended_contact_pct','hand_object_peak_mm','whole_object_peak_mm','self_peak_mm','arm_jerk_rad_s3','refinement_seconds'])
    doc+='<p>Physical F1: tip collision shape within 2 mm and penetration ≤1 mm, compared with fixed source fingertip proximity labels (&lt;30 mm). Intended contact additionally requires the original-mesh regional gap ≤2 mm and all distal-tip vertices on the intended side within 1 mm. Patch gap uses original mesh surfaces. Penetration maxima are per-clip maxima averaged across clips; separate combined duration/depth columns retain the historical violating-frame definition. No thresholds changed.</p>'
    doc+='<h2>Collider accuracy</h2><p>The 32-piece limit does not guarantee the requested concavity or surface accuracy. Failing the sampled 0.5 mm check means these colliders are experimental, not certified contact surfaces. Automatic convex components are accepted only after a separate bidirectional surface check. Source-only winding repairs never move vertices or close openings.</p>'
    doc+=table(geometry,['sequence','pieces','method','preparation_seconds','timing_scope','median_error_mm','p95_error_mm','maximum_error_mm','sampled_half_mm_pass'])
    doc+=table(records,['sequence','dataset','split','variant','source_contact_frames','patch_gap_mm','physical_f1_pct','intended_contact_pct','hand_object_peak_mm','leg_object_peak_mm','arm_jerk_rad_s3','refinement_seconds'])
    if (OUT/'table_geometry_comparison.json').exists():
        doc+='<section><h2>Generic collider improvement: table</h2><p>Automatic connected-component hulls, accepted by the same 0.5 mm sampled surface check. Nine pieces instead of 18; all objective weights unchanged. <a href="table_geometry_comparison.json">Both motions scored with the same validated colliders</a>.</p>'
        seq='push_table_forward_with_both_hands-039'
        for suffix in ('','_hands'):doc+=f'<video controls preload="metadata" src="renders_table_geometry/{seq}{suffix}.mp4"></video>'
        doc+='</section>'
    for meta in sorted((OUT/'renders').glob('*.json')):
        seq=meta.stem
        if not (OUT/'renders'/f'{seq}.mp4').exists():continue
        panels=read(meta).get('panels',[])
        labels=['Human demonstration' if 'human' in p else LABELS.get(p,p) for p in panels]
        doc+='<section><h2>'+html.escape(seq)+'</h2><p>'+html.escape(' / '.join(labels))+'. Identical frames, camera and object scale. Close-up below. '
        doc+=f'<a href="renders/{seq}.mp4">Whole-body MP4</a>; <a href="renders/{seq}_hands.mp4">hands MP4</a>.</p>'
        for suffix in ('','_hands'):doc+=f'<video controls preload="metadata" src="renders/{seq}{suffix}.mp4"></video>'
        doc+='</section>'
    doc+='<h2>Failures</h2><pre>'+html.escape(__import__('json').dumps(failures,indent=2))+'</pre>'
    doc+='<h2>Recovered initial failures</h2><pre>'+html.escape(__import__('json').dumps(recovery,indent=2))+'</pre>'
    (OUT/'index.html').write_text(doc)
    print('Reported',len(records),'completed evaluated runs;',len(failures),'failed stages',flush=True)
if __name__=='__main__':main()
