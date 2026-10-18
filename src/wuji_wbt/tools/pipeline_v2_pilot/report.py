"""Generate a matched pilot report without changing any evaluation definitions."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import html

WORK=Path(__file__).resolve().parents[4]
OUT=WORK.parent/'holosoma/eval_out/ref_candidates/pipeline_v2_pilot_20261018'

def read(p):return json.loads(p.read_text())
def avg(values):
    a=[v for v in values if v is not None]
    return statistics.mean(a) if a else None
def fmt(x):return 'N/A' if x is None else f'{x:.2f}'
def table(columns,rows):
    return '<table><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in columns)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'

cohort=read(OUT/'cohort.json')['sequences']
completed=[]
for seq in cohort:
    d=OUT/'clips'/seq
    if (d/'completed.json').exists():
        completed.append(dict(sequence=seq,metrics=read(d/'metrics.json'),timing=read(d/'completed.json'),run=read(d/'run.json')))
metrics=['contact_precision_pct','contact_recall_pct','contact_f1_pct',
         'contact_f1_penetration_gated_pct','contact_f1_penetration_gated5mm_pct','contact_location_gap_mm',
         'primary_tip_error_mm','secondary_tip_error_mm','palm_normal_error_deg',
         'combined_pen_duration_pct','combined_pen_conditional_depth_cm',
         'hand_object_pen_mean_mm','hand_object_pen_duration_gt20mm_pct',
         'self_pen_nonadjacent_duration_gt10mm_pct','ground_pen_duration_gt10mm_pct',
         'body_bone_direction_error_deg','body_joint_position_error_mm','body_jerk_mean_m_s3','foot_skating_mean_cm_s']
aggregate={v:{k:dict(mean=avg([r['metrics'][v].get(k) for r in completed]),
                     n=sum(r['metrics'][v].get(k) is not None for r in completed))
              for k in metrics} for v in ('v1','v2')}
timings=[]
for r in completed:
    old=r['timing']['baseline_stage_seconds'];run=r['run'];tim=r['timing']
    parts=[old.get(k) for k in ('part1','part2','part3','convert50hz')]
    v1ret=sum(parts) if all(v is not None for v in parts) else None
    timings.append(dict(sequence=r['sequence'],v1_body_seconds=old['part1'],
        v1_floating_hands_seconds=old['part2'],v1_arm_refinement_seconds=old['part3'],
        v1_retarget_and_export_seconds=v1ret,
        v2_body_seconds=tim['measured_body_seconds'],
        v2_preparation_seconds=run['preparation_seconds'],
        v2_quick_fingers_seconds=run['finger_initialization_seconds'],
        v2_joint_refinement_seconds=run['joint_refinement_seconds'],
        v2_export_seconds=tim['measured_export_seconds'],
        v2_retarget_and_export_seconds=tim['total_sequential_retarget_export_seconds'],
        observed_total_speedup=v1ret/tim['total_sequential_retarget_export_seconds'] if v1ret is not None else None,
        repeated_body_max_difference=tim['repeated_body_max_difference']))
summary=dict(cohort_size=len(cohort),completed=len(completed),frames=sum(r['metrics']['frames'] for r in completed),
    aggregate=aggregate,per_sequence_timings=timings,
    mean_v1_retarget_export_seconds=avg([t['v1_retarget_and_export_seconds'] for t in timings]),
    mean_v2_retarget_export_seconds=avg([t['v2_retarget_and_export_seconds'] for t in timings]),
    all_frozen_body_checks_passed=all(r['metrics']['unchanged_body_exact'] for r in completed),
    all_source_contact_masks_identical=all(r['metrics']['unchanged_source_contact_mask'] for r in completed),
    all_solver_calls_succeeded=all(r['run']['solver_failures']==0 for r in completed),
    numerical_checks=read(OUT/'numerical_checks.json'),
    caveats=['Development clips, not a held-out benchmark.',
             'The two additional v1 outputs are existing historical references, unchanged per user request; organizer v1 had missing arm-stage object-collision lookup.',
             'HUMOTO uses Mixamo landmarks, not fitted MANO parameters.',
             'Soft collision slack permits residual violations; quality is measured on the direct 30-Hz export before tracker resampling.',
             'V1 historical times and v2 measured times share a workstation but have different concurrent load.',
             'Retarget+export excludes input extraction, rendering, evaluation, and development tests.',
             'Body is rerun for timing; exact archived body is used for paired quality comparison.',
             'Same existing CoACD geometry and inferred source-contact metrics; not an exact paper reproduction.'])
for filename,key in [('output_checks.json','output_checks'),('residual_collisions.json','residual_collisions'),
                     ('media_checks_human.json','human_media_checks'),('baseline_reuse_checks.json','baseline_reuse_checks')]:
    if (OUT/filename).exists():summary[key]=read(OUT/filename)
summary['mean_stage_seconds']={k:avg([t[k] for t in timings]) for k in (
    'v1_body_seconds','v1_floating_hands_seconds','v1_arm_refinement_seconds',
    'v2_body_seconds','v2_quick_fingers_seconds','v2_joint_refinement_seconds','v2_export_seconds')}
summary['timing_counts']={v:sum(t[f'{v}_retarget_and_export_seconds'] is not None for t in timings) for v in ('v1','v2')}
summary['initial_four_results']='initial_four/summary.json'
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
csvrows=[]
for r in completed:
    for v in ('v1','v2'):
        csvrows.append(dict(sequence=r['sequence'],variant=v,frames=r['metrics']['frames'],
                           **{k:r['metrics'][v].get(k) for k in metrics}))
if csvrows:
    for filename,rows in [('metrics.csv',csvrows),('timings.csv',timings)]:
        with (OUT/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
labels=['Contact precision (%) ↑','Contact recall (%) ↑','Contact F1 at 30 mm (%) ↑',
        'Contact F1 with strict 1 mm hand-penetration gate (%) ↑',
        'Contact F1 with 5 mm hand-penetration gate (%) ↑','Contact-point gap (mm) ↓',
        'Thumb/index error (mm) ↓','Other finger error (mm) ↓','Palm normal error (deg) ↓',
        'Combined penetration duration (%) ↓','Conditional frame-max penetration depth (cm) ↓',
        'Mean hand/object penetration (mm) ↓','Hand/object penetration >20 mm (%) ↓',
        'Self penetration >10 mm (%) ↓','Ground penetration >10 mm (%) ↓',
        'Body bone-direction error (deg) ↓','Body landmark error (mm) ↓','Body jerk (m/s³) ↓','Mean foot skating (cm/s) ↓']
quality=[[label]+[fmt(aggregate[v][k]['mean'])+f" (n={aggregate[v][k]['n']})" for v in ('v1','v2')]
         for label,k in zip(labels,metrics)]
page='''<!doctype html><meta charset="utf-8"><title>Wuji HUMOTO pipeline v2 pilot</title>
<style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;line-height:1.55}
table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #ccd;padding:8px;text-align:right}
td:first-child,th:first-child{text-align:left}th{background:#edf3f9}video,img{max-width:100%}aside{background:#fff3da;padding:15px}</style>
<h1>HUMOTO + Wuji: pipeline v1 versus v2</h1>'''
page+=f"<p>{len(completed)}/{len(cohort)} complete clips, {summary['frames']:,} frames. Equal weight per clip; undefined conditional metrics excluded with counts saved in JSON.</p>"
page+='''<aside>V2 initializes fingers from source anatomical landmarks, jointly refines both arms and fingers,
then exports them directly. Base, legs, waist and object are unchanged. This is a development pilot.
HUMOTO supplies Mixamo landmarks, not MANO coefficients. Relevant collision constraints include penalized
slack; the saved motion is evaluated for actual residual penetration.</aside><h2>Matched quality</h2>'''
page+='''<p>The two added clips use the existing V1 references exactly as requested; no V1 stage was rerun.
The organizer's historical arm stage omitted object-collision lookup. Its result therefore compares against
that historical reference, not the corrected 120-clip implementation. <a href="initial_four/summary.json">Original four-clip results</a>.</p>'''
page+=table(['Metric','V1 existing SEPARATE','V2 direct joint refinement'],quality)
page+='<h2>Per-clip tradeoffs</h2>'
page+=table(['Clip','F1 V1 → V2 (%)','Penetration duration V1 → V2 (%)',
             'Thumb/index error V1 → V2 (mm)','Palm error V1 → V2 (deg)'],
            [[r['sequence']]+[fmt(r['metrics']['v1'][k])+' → '+fmt(r['metrics']['v2'][k]) for k in
              ('contact_f1_pct','combined_pen_duration_pct','primary_tip_error_mm','palm_normal_error_deg')]
             for r in completed])
if len(completed)==len(cohort):
    page+='''<aside><strong>Interpretation:</strong> consult both aggregate and per-clip results; gains are not universal.
The clothes-rack clip loses contact recall and has worse fingertip/palm alignment and body direction.
The strict 1 mm penetration-gated F1 also worsens: v2 often rests just beyond its 1 mm soft tolerance.
This is not a claim of collision-free motion or demonstrated grasp stability.</aside>'''
if 'residual_collisions' in summary:
    page+='''<p><a href="residual_collisions.json">Per-frame residual collision pairs</a>. Fixed leg/hip
collisions cannot be repaired by an optimizer that must preserve those body joints.</p>'''
page+='''<p>Combined penetration: object &gt;20 mm OR ground/self &gt;10 mm, counted once per frame.
Depth averages raw frame maxima over violating frames, then equally over defined clips. Contact labels,
source targets, collision geometry and all other evaluation definitions are unchanged from the prior evaluation.</p>
<p>Proximity F1 uses a 30 mm threshold and is not a physical grasp success rate. The stricter
1 mm penetration gate can reject contacts just beyond the solver's 1 mm soft target. The 5 mm
gate and raw depths are also shown; no threshold was changed after observing these results.</p>
<h2>Measured stage times</h2>'''
page+=table(['Clip','V1 body (s)','V1 floating hands (s)','V1 arms (s)','V1 total incl. export (s)',
             'V2 body (s)','V2 finger init (s)','V2 joint solve (s)','V2 total incl. export (s)','Observed speedup'],
            [[t['sequence']]+[fmt(t[k]) for k in ('v1_body_seconds','v1_floating_hands_seconds','v1_arm_refinement_seconds',
              'v1_retarget_and_export_seconds','v2_body_seconds','v2_quick_fingers_seconds','v2_joint_refinement_seconds',
              'v2_retarget_and_export_seconds','observed_total_speedup')] for t in timings])
page+='''<p>Totals sum sequential body + refinement-process + 50 Hz export time; input extraction, evaluation,
rendering and development work are excluded. The pilot runs body timing and paired refinement concurrently.
Historical V1 ran under a different concurrent load; speedup is an observed comparison, not a controlled hardware benchmark.</p>
<p>Historical V1 stage runtimes for the two added clips are unavailable and shown as N/A, never zero.
The V1 timing mean uses four clips; the V2 timing mean uses all completed clips. Do not divide those means
to claim a six-clip speedup.</p>
<p><a href="summary.json">Full results</a> | <a href="metrics.csv">Per-clip metrics</a> |
<a href="timings.csv">Per-clip timings</a> | <a href="protocol.json">Protocol</a> |
<a href="numerical_checks.json">Numerical checks</a></p><h2>Visual comparison</h2>'''
page+='<p><a href="output_checks.json">Export and invariance checks</a> | <a href="media_checks_human.json">Human/video checks</a> | <a href="baseline_reuse_checks.json">Unchanged V1 checks</a> | <a href="source_provenance.json">Source provenance</a> | <a href="v2_source_snapshot.tar.gz">Source snapshot</a></p>'
for seq in cohort:
    page+='<h3>'+html.escape(seq)+'</h3>'
    folder='renders_human'
    if (OUT/folder/f'{seq}.json').exists():
        page+=f'<p>Left: original human demonstration, center: existing V1, right: V2. Human and object use the same canonical scene scale; 15 FPS display samples the full 30 Hz sequence.</p><video controls preload="metadata" src="{folder}/{seq}.mp4"></video><p><a href="{folder}/{seq}.mp4">Three-panel MP4</a> | <a href="{folder}/{seq}.png">Full-body PNG</a> | <a href="{folder}/{seq}_hands.png">Hand close-up PNG</a></p>'
    else:
        page+='<p>Human/V1/V2 rendering pending.</p>'
(OUT/'index.html').write_text(page)
print(json.dumps({k:v for k,v in summary.items() if k not in ('per_sequence_timings','numerical_checks','caveats')},indent=2))
