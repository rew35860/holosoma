"""Macro and paired summaries of the fixed objective ablation; no metric changes."""
import csv
import html
import statistics
from config import *

METRICS = [
    ('contact_f1_pct','Contact F1 (%) ↑',True),
    ('contact_recall_pct','Contact recall (%) ↑',True),
    ('contact_precision_pct','Contact precision (%) ↑',True),
    ('contact_f1_penetration_gated_pct','F1, strict 1 mm penetration gate (%) ↑',True),
    ('contact_f1_penetration_gated5mm_pct','F1, 5 mm penetration gate (%) ↑',True),
    ('contact_location_gap_mm','Contact-point gap (mm) ↓',False),
    ('combined_pen_duration_pct','Combined penetration duration (%) ↓',False),
    ('combined_pen_conditional_depth_cm','Conditional frame-max depth (cm) ↓',False),
    ('hand_object_pen_mean_mm','Mean hand/object depth (mm) ↓',False),
    ('hand_object_pen_max_mm','Mean of clip-max hand/object depths (mm) ↓',False),
    ('primary_tip_error_mm','Thumb/index error (mm) ↓',False),
    ('secondary_tip_error_mm','Other fingertip error (mm) ↓',False),
    ('palm_normal_error_deg','Palm normal error (deg) ↓',False),
    ('body_bone_direction_error_deg','Body direction error (deg) ↓',False),
    ('body_joint_position_error_mm','Body landmark error (mm) ↓',False),
    ('link_velocity_direction_error_deg','Link velocity direction error (deg) ↓',False),
    ('body_jerk_mean_m_s3','Body jerk (m/s³) ↓',False),
    ('object_relative_fingertip_velocity_error_cm_s','Object-relative fingertip velocity error (cm/s) ↓',False),
    ('relative_hand_orientation_increment_error_deg','Relative hand orientation increment error (deg) ↓',False),
    ('foot_skating_mean_cm_s','Foot skating (cm/s), fixed legs',False),
]
TIMES = ['preparation_seconds','finger_initialization_seconds','joint_refinement_seconds',
         'clarabel_solver_seconds','measured_refinement_process_seconds','total_subproblems',
         'line_search_stopped_frames','evaluation_process_seconds']

def mean(a):
    a=[x for x in a if x is not None]
    return statistics.mean(a) if a else None

def fmt(x):
    return 'N/A' if x is None else f'{x:.2f}'

def table(cols, rows):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in cols)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</table></div>'

def main():
    status=read(OUT/'status.json')
    records={v:{} for v in REMOVALS}
    for v in REMOVALS:
        for seq in COHORT:
            d=OUT/'variants'/v/'clips'/seq
            if (d/'completed.json').exists():
                records[v][seq]=dict(timing=read(d/'completed.json'),metrics=read(d/'metrics.json')['v2'],
                                     checks=read(d/'checks.json'))
    aggregate={v:{k:dict(mean=mean([r['metrics'].get(k) for r in records[v].values()]),
                             n=sum(r['metrics'].get(k) is not None for r in records[v].values()))
                  for k,_,_ in METRICS} for v in REMOVALS}
    shared=[seq for seq in COHORT if all(seq in records[v] for v in REMOVALS)]
    matched={v:{k:dict(mean=mean([records[v][s]['metrics'].get(k) for s in shared]),
                       n=sum(records[v][s]['metrics'].get(k) is not None for s in shared))
                for k,_,_ in METRICS} for v in REMOVALS}
    timing={v:{k:mean([r['timing'][k] for r in records[v].values()]) for k in TIMES} for v in REMOVALS}
    common_timing={v:{k:mean([records[v][s]['timing'][k] for s in shared]) for k in TIMES} for v in REMOVALS}
    paired={}
    for v in REMOVALS:
        if v=='full':continue
        common=[seq for seq in COHORT if seq in records[v] and seq in records['full']]
        paired[v]={}
        for k,_,higher in METRICS:
            ds=[records[v][s]['metrics'][k]-records['full'][s]['metrics'][k] for s in common
                if records[v][s]['metrics'].get(k) is not None and records['full'][s]['metrics'].get(k) is not None]
            paired[v][k]=dict(mean_delta=mean(ds),n=len(ds),
                improved=sum(d>1e-8 if higher else d < -1e-8 for d in ds),
                worsened=sum(d < -1e-8 if higher else d>1e-8 for d in ds),
                unchanged=sum(abs(d)<=1e-8 for d in ds))
    pooled={v:dict(frames=sum(r['timing']['frames'] for r in records[v].values()),
        violating_frames=sum(r['metrics']['penetration_violating_frames'] for r in records[v].values()),
        violating_frame_depth_sum_m=sum(r['metrics']['violating_frame_depth_sum_m'] for r in records[v].values()))
        for v in REMOVALS}
    for p in pooled.values():
        p['frame_weighted_penetration_duration_pct']=100*p['violating_frames']/p['frames'] if p['frames'] else None
        p['pooled_conditional_frame_max_depth_cm']=100*p['violating_frame_depth_sum_m']/p['violating_frames'] if p['violating_frames'] else None
    worst={v:dict(max_hand_object_depth_mm=max([r['metrics']['hand_object_pen_max_mm'] for r in records[v].values()],default=None),
                  max_primary_error_mm=max([r['metrics']['primary_tip_error_mm'] for r in records[v].values()],default=None),
                  min_contact_f1_pct=min([r['metrics']['contact_f1_pct'] for r in records[v].values()],default=None))
           for v in REMOVALS}
    summary=dict(state=status['state'],completed={v:len(records[v]) for v in REMOVALS},
        sequences=COHORT,aggregate=aggregate,mean_timing=timing,paired_delta_vs_full=paired,
        shared_successful_sequences=shared,common_cohort_aggregate=matched,
        common_cohort_mean_timing=common_timing,
        all_planned_runs_succeeded=all(len(records[v])==len(COHORT) for v in REMOVALS),
        pooled_penetration=pooled,worst_clip_values=worst,
        all_checks_passed=all(r['checks']['passed'] for rs in records.values() for r in rs.values()),
        full_trajectory_max_difference=max([r['checks']['full_v2_trajectory_max_difference'] for r in records['full'].values()],default=None),
        note='Macro means weight clips equally. Paired deltas use common completed clips; conditional metrics omit undefined clips. Full results need all six clips per variant.')
    write(OUT/'summary.json',summary)
    rows=[];tr=[]
    for v,rs in records.items():
        for seq,r in rs.items():
            rows.append(dict(variant=v,sequence=seq,**{k:r['metrics'].get(k) for k,_,_ in METRICS}))
            tr.append(dict(variant=v,sequence=seq,**{k:r['timing'][k] for k in TIMES}))
    for filename,values in [('metrics.csv',rows),('timings.csv',tr)]:
        if values:
            with (OUT/filename).open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    page='''<!doctype html><meta charset="utf-8"><title>Wuji V2 objective ablation</title>
<style>body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 20px;line-height:1.5}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{border:1px solid #ccd;padding:7px;text-align:right}
td:first-child,th:first-child{text-align:left}th{background:#edf3f9}aside{background:#fff3da;padding:14px}a{color:#175d94}</style>
<h1>HUMOTO / Wuji: V2 objective ablation</h1>'''
    page+=f'<p>Status: <b>{html.escape(status["state"])}</b>. {status["completed"]}/{status["total"]} refinement runs complete; six complete sequences per variant.</p>'
    page+='''<aside>Only the specified objective weights change. Same saved Omni body, quick finger initialization,
source demonstration, collision geometry/constraints, solver settings and evaluation definitions.
All five additions removed means no separate fingertip tracking, source-contact attraction, palm alignment,
finger initialization prior, or Omni arm prior (both contact and non-contact). Interaction mesh (including its
original fingertip emphasis), bone matching, temporal smoothing and constraints remain.
This is a V2 core model, not a rerun of V1.</aside>'''
    page+='<p><a href="protocol.json">Frozen protocol</a> | <a href="summary.json">Summary JSON</a> | <a href="metrics.csv">Per-clip metrics CSV</a> | <a href="timings.csv">Timing CSV</a> | <a href="input_manifest.json">Input hashes</a> | <a href="../pipeline_v2_pilot_20261018/index.html">Original pilot and human videos</a></p>'
    if (OUT/'interpretation.json').exists():
        interpretation=read(OUT/'interpretation.json')
        page+='<h2>Findings</h2>'+''.join('<p>'+html.escape(t)+'</p>' for t in interpretation['findings'])
        page+='<p><a href="settings/no_fingertip.json">Tested simpler candidate settings</a>. Production defaults are unchanged.</p>'
    page+='<h2>Settings changed</h2>'+table(['Variant','Clips completed','Settings changed'],[[LABELS[v],str(len(records[v]))+'/6',html.escape(str(REMOVALS[v]))] for v in REMOVALS])
    page+='<h2>Quality: equal weight per sequence</h2>'+table(['Metric']+[LABELS[v] for v in REMOVALS],
        [[label]+[fmt(aggregate[v][k]['mean'])+f" (n={aggregate[v][k]['n']})" for v in REMOVALS] for k,label,_ in METRICS])
    if len(shared)<len(COHORT):
        page+='<aside>Some variants are incomplete or failed. Available-clip means above may cover different clips. Consult completion counts and paired deltas; missing clips are never assigned a zero metric.</aside>'
        page+=f'<h2>Quality on the shared completed subset ({len(shared)} clips)</h2><p>This secondary table uses identical clips for every variant. It does not erase failed runs or measure their quality. Shared clips: '+html.escape(', '.join(shared))+'.</p>'
        page+=table(['Metric']+[LABELS[v] for v in REMOVALS],
            [[label]+[fmt(matched[v][k]['mean'])+f" (n={matched[v][k]['n']})" for v in REMOVALS] for k,label,_ in METRICS])
    page+='''<p>Combined penetration duration: object &gt;20 mm OR ground/self &gt;10 mm, one event per frame.
Conditional depth: raw frame maximum averaged over violating frames, then over defined clips (n shown).
F1 is inferred fingertip proximity at 30 mm. Strict 1 mm and tolerant 5 mm penetration-gated F1 prevent proximity
alone from concealing penetration. None of these scores establishes dynamic grasp stability.</p>'''
    page+='<h2>Worst-case checks</h2>'+table(['Variant','Lowest clip contact F1 (%)','Largest clip thumb/index error (mm)','Deepest hand/object collision across frames (mm)'],
        [[LABELS[v]]+[fmt(worst[v][k]) for k in ('min_contact_f1_pct','max_primary_error_mm','max_hand_object_depth_mm')] for v in REMOVALS])
    page+='<h2>Paired change from full V2</h2><p>Cells show variant minus full V2, then improved/worsened clip counts. Use complete paired rows for conclusions.</p>'
    paired_keys=['contact_f1_pct','combined_pen_duration_pct','primary_tip_error_mm','palm_normal_error_deg','body_jerk_mean_m_s3']
    page+=table(['Variant']+[next(label for k,label,_ in METRICS if k==key) for key in paired_keys],
        [[LABELS[v]]+[fmt(paired[v][k]['mean_delta'])+f" ({paired[v][k]['improved']} better / {paired[v][k]['worsened']} worse; n={paired[v][k]['n']})" for k in paired_keys] for v in paired])
    page+='<h2>Refinement runtime per clip</h2>'+table(['Variant','Prepare (s)','Finger init (s)','Joint solve (s)','Clarabel alone (s)','Whole refinement process (s)','Subproblems'],
        [[LABELS[v]]+[fmt(timing[v][k]) for k in TIMES[:6]] for v in REMOVALS])
    if len(shared)<len(COHORT):
        page+=f'<p>Matched runtime on the same {len(shared)} completed clips:</p>'+table(['Variant','Joint solve (s)','Whole refinement process (s)'],
            [[LABELS[v],fmt(common_timing[v]['joint_refinement_seconds']),fmt(common_timing[v]['measured_refinement_process_seconds'])] for v in REMOVALS])
    page+='''<p>Body trajectories are reused. Times exclude body retargeting, evaluation, visualization and tracker export.
Full V2 is rerun under the same CPU limits. One observation per clip/variant; concurrent workstation load remains a confound.
Zero-weight residuals still execute their geometry calculations, so this measures the current implementation,
not the best achievable runtime after deleting unused code.</p><h2>Per-clip comparison</h2>'''
    detail=['contact_f1_pct','combined_pen_duration_pct','primary_tip_error_mm','palm_normal_error_deg','body_bone_direction_error_deg']
    for seq in COHORT:
        page+='<h3>'+html.escape(seq)+'</h3>'+table(['Variant']+[next(label for k,label,_ in METRICS if k==key) for key in detail],
            [[LABELS[v]]+[fmt(records[v][seq]['metrics'].get(k)) if seq in records[v] else 'pending' for k in detail] for v in REMOVALS])
    if status['errors']:
        page+='<h2>Failed runs retained</h2>'+table(['Variant','Clip','Evidence'],
            [[e['variant'],e['sequence'],f'<a href="variants/{e["variant"]}/clips/{e["sequence"]}/refinement.log">Original solve log</a>'] for e in status['errors']])
        page+='<p>Failure status is separate from quality on completed clips. Solver tolerances or objectives are not relaxed to turn a failure into a success.</p>'
    if (OUT/'retry_summary.json').exists():
        page+='<p><a href="retry_summary.json">Independent unchanged-settings retry results and diagnostic locations</a>.</p>'
    if (OUT/'ablation_summary.png').exists():
        page+='<h2>Shared-cohort plot</h2><p><a href="ablation_summary.png">PNG</a> | <a href="ablation_summary.pdf">PDF</a></p><img style="max-width:100%" src="ablation_summary.png">'
    page+='''<p>Interpretation: single-term removal effects depend on the other active terms. The all-removed variant
tests their joint effect. A larger F1 alone is not sufficient to choose a model; inspect penetration, alignment,
body deviation, jerk and per-clip failures. This development set does not establish generalization.</p>'''
    (OUT/'index.html').write_text(page)

if __name__=='__main__':
    main()
