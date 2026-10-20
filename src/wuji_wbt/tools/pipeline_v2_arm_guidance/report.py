"""Paired metrics and local visual audit; original evaluation stays unchanged."""
import csv
import html
import numpy as np
from config import *


def silhouette_gallery():
    """Link offline hover views without changing any experiment or metric data."""
    cards=[]
    for seq in COHORT:
        folder=OUT/'graphs'/seq
        if not (folder/'silhouettes.json').exists():continue
        for frame in read(folder/'silhouettes.json')['frames']:
            stem=f'{seq}/frame_{frame["frame"]:03d}'
            cards.append(f'<section><h2>{html.escape(seq)} — frame {frame["frame"]}</h2>'
                f'<p><a href="{stem}_silhouettes_hover.html">Open body + hand graphs with hover labels</a> · '
                f'<a href="{stem}_hands_silhouettes_hover.html">Open hand close-ups with hover labels</a></p>'
                f'<a href="{stem}_silhouettes_hover.html"><img loading="lazy" src="{stem}_silhouettes.png"></a>'
                f'<p><a href="{stem}_silhouettes.png">Full-view PNG</a> · '
                f'<a href="{stem}_hands_silhouettes.png">Hand close-up PNG</a> · '
                f'<a href="{stem}_weights.png">Neighbour weights</a> · '
                f'<a href="{seq}/arm_gradient.png">Local objective gradients</a></p></section>')
    if not cards:return False
    (OUT/'graphs'/'silhouettes.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Interaction graphs with human and object silhouettes</title>'
        '<style>body{font:16px system-ui;max-width:1400px;margin:28px auto;padding:0 20px;color:#173246}'
        'img{max-width:100%}section{border-top:1px solid #ccd5dd;margin:25px 0}a{color:#155f9b}</style>'
        '<h1>Body and coupled-hand interaction graphs</h1><p><a href="../index.html">Back to comparison report</a></p>'
        '<p>Blue translucent surface: cached human demonstration. Tan surface: the original object triangle mesh. '
        'Orange links: body/hand to object samples. Thick dark or finger-colored lines are anatomical guides only. '
        'Top row: human graph. Bottom row: saved robot graph over the same human reference ghost. '
        'The body graph is from the Omni body pass; the hand graphs are from the coupled arm/finger pass.</p>'
        '<p>Open a labeled view and hover over a node for its landmark name, graph index and world coordinates. '
        'Graph arrays, objectives, optimized poses and evaluation definitions are unchanged. '
        'Edge widths represent neighbour weights; they do not measure causal influence on the final pose.</p>'
        +''.join(cards))
    return True


def main(visuals_only=False):
    rows=[]
    for v in REMOVALS:
        for seq in COHORT:
            dest=OUT/'variants'/v/'clips'/seq
            if not (dest/'completed.json').exists():continue
            row=dict(variant=v,sequence=seq,**read(dest/'metrics.json')['v2'])
            if (dest/'comfort.json').exists():row.update(read(dest/'comfort.json'))
            if (dest/'collision_audit.json').exists():
                audit=read(dest/'collision_audit.json')
                for group in ('moving_arm_hand_self','other_robot_self','screened_hand_object'):
                    row.update({group+'_'+k:value for k,value in audit[group].items()})
            row['joint_refinement_seconds']=read(dest/'completed.json')['joint_refinement_seconds']
            rows.append(row)
    shared=[s for s in COHORT if all(any(r['sequence']==s and r['variant']==v for r in rows) for v in REMOVALS)]
    aggregates={}
    for v in REMOVALS:
        selected=[r for r in rows if r['variant']==v and r['sequence'] in shared]
        keys=set.intersection(*(set(r) for r in selected)) if selected else set()
        numeric=[k for k in keys if all(isinstance(r[k],(int,float)) and not isinstance(r[k],bool) for r in selected)]
        aggregates[v]=dict(successful_clips=sum(r['variant']==v for r in rows),paired_clips=len(selected),
            clip_equal_means={k:float(np.mean([r[k] for r in selected])) for k in sorted(numeric)})
    summary=dict(completed=len(rows),planned=18,shared_successful_sequences=shared,aggregates=aggregates,
        aggregation='Equal clip means over clips completed by all three variants. Failures separately counted; no imputation.',
        changed='Only arm_prior_contact: 0.002 / 0.0005 / 0. All keep arm_prior_no_contact=0.05.',
        penetration_caution='Frozen native MuJoCo evaluator retained. Organizer baseline frame 140 includes an independently disproven 34 mm hand-object penetration. Raw penetration numbers are not fully validated collision measurements.',rows=rows)
    summary['failures']=read(OUT/'status.json').get('errors',[]) if (OUT/'status.json').exists() else []
    summary['unchanged_retries']=[read(p) for p in (OUT/'retries').glob('*/*/retry.json')]
    if len(rows)==17 and len(shared)==5:
        summary['interpretation']=[
            'No consistent improvement in robot comfort: common-five contact F1 changes by less than 0.1 percentage point, while mean minimum arm-limit margin falls and acceleration/jerk increase.',
            'The basket benefits in limit margin but becomes less smooth. On organizer, zero contact prior increases near-limit contacting hand-frames from 2.63% to 27.75%, mainly left wrist yaw.',
            'High 30 mm contact F1 is not strict physical contact: common-five 1 mm penetration-gated F1 remains 0.0833%.',
            'Default V2 is unchanged. Relaxing a human-reference prior does not supply an explicit robot-comfort preference; body/legs remain fixed in this experiment.']
    if not visuals_only:write(OUT/'summary.json',summary)
    if rows and not visuals_only:
        keys=['variant','sequence']+sorted(set.union(*(set(r) for r in rows))-{'variant','sequence','definitions'})
        with (OUT/'per_sequence.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
    columns=[('contact_f1_pct','Contact F1 % ↑'),('contact_recall_pct','Recall % ↑'),
        ('fingertip_surface_distance_on_contact_mm','Surface gap mm ↓'),('contact_location_gap_mm','Source point gap mm ↓'),
        ('contact_f1_penetration_gated_pct','F1 with 1 mm gate % ↑'),('hand_object_pen_mean_mm','Raw hand depth mm ↓'),
        ('screened_hand_object_mean_mm','Screened hand depth mm ↓'),
        ('moving_arm_hand_self_duration_gt1mm_pct','Moving arm/hand self >1 mm % ↓'),
        ('moving_arm_hand_self_max_mm','Moving arm/hand self max mm ↓'),
        ('contact_arm_min_margin_mean_deg','Arm minimum margin mean ° ↑'),('contact_arm_min_margin_p05_deg','Arm margin p05 ° ↑'),
        ('contact_hand_near_limit_5deg_pct','Arm near limit % ↓'),('contact_arm_velocity_rms_rad_s1','Arm speed RMS rad/s'),
        ('contact_arm_acceleration_rms_rad_s2','Arm accel RMS rad/s² ↓'),('contact_arm_jerk_rms_rad_s3','Arm jerk RMS rad/s³ ↓'),
        ('arm_change_from_full_rms_deg','Arm change RMS °')]
    def table(rs,labels):
        head='<tr><th>Variant</th>'+''.join(f'<th>{html.escape(label)}</th>' for _,label in columns)+'</tr>'
        body=''
        for r,label in zip(rs,labels):
            body+='<tr><td>'+html.escape(label)+'</td>'+''.join('<td>'+('—' if r.get(k) is None else f'{r[k]:.3f}')+'</td>' for k,_ in columns)+'</tr>'
        return '<div class="scroll"><table>'+head+body+'</table></div>'
    page=['<!doctype html><meta charset="utf-8"><title>Contact-only arm guidance study</title><style>body{font:16px system-ui;margin:28px;color:#182633;max-width:1600px}table{border-collapse:collapse;font-size:13px}th,td{padding:9px;border:1px solid #ccd5dd}th{background:#edf3f8}td{text-align:right}.scroll{overflow:auto}img,video{max-width:100%;display:block;margin:14px 0}a{color:#155f9b}.note{padding:16px;background:#fff5df}details{margin:20px 0}</style>',
        '<h1>Does weaker arm guidance improve robot posture while preserving contact?</h1>',
        f'<p>{len(rows)}/18 trials complete. Paired aggregate uses {len(shared)} of six clips. Same saved Omni body pass; coupled solver defaults unchanged.</p>',
        '<p>During contact only, the Omni arm prior is 0.002 (current), 0.0005 (quarter), or 0. Outside contact it stays 0.05. Every hand objective, collision parameter, joint/rate/trust limit, input and initializer stays unchanged.</p>',
        '<p>Source contact is fingertip distance &lt;30 mm from the original object mesh. A contacting hand has at least one contacting fingertip. Comfort diagnostics use contacting hand-frames; minimum margin is over seven arm/wrist joints. Derivative stencils stay entirely within contact, at 30 Hz without filtering. Larger margin and lower acceleration/jerk are proxies; they do not establish subjective naturalness or dynamic grasp stability.</p>',
        '<p class="note">'+html.escape(summary['penetration_caution'])+' Original evaluation definitions and raw outputs remain available. Check geometry before treating large depths as real collisions.</p>',
        '<p>Supplementary collision audit: hand-object distances above 5 mm are independently checked for convex separation; certified false positives are excluded only from the explicitly named screened diagnostic. Smaller depths remain native measurements. Moving arm/hand self contacts are separated from other robot contacts using the solver collision-pair list. These additional definitions do not replace the frozen metrics.</p>',
        '<h2>Paired clip-equal means</h2>',table([aggregates[v]['clip_equal_means'] for v in REMOVALS],[f'{LABELS[v]} ({aggregates[v]["successful_clips"]}/6 solved)' for v in REMOVALS]),
        '<p><a href="per_sequence.csv">Full metrics CSV</a> · <a href="summary.json">Summary JSON</a> · <a href="protocol.json">Protocol</a> · <a href="status.json">Solver status</a></p>',
        '<h2 id="interaction-graphs">What the graphs mean</h2><p>The body pass uses 15 landmarks plus object samples, uniform neighbour weights and mesh weight 10. The coupled pass uses two separate 21-landmark hand graphs plus the same object samples, exp(-30 × edge length) neighbour weights, normalized mesh weight 500/N, and fingertip row weight 10. No shoulder or elbow nodes. Edges and weights are fixed within a frame and rebuilt from the source at the next frame; the contact flag does not alter them.</p>',
        '<p>Orange edges connect human/robot nodes to object samples; blue connects body/hand nodes; grey connects object samples. Source above, saved robot below. Matrices show directed neighbour weights. Gradient bars measure local sensitivity at the same baseline pose, not causal percentages. Removing the explicit arm prior leaves hand-position/orientation guidance, temporal regularization, constraints and initialization.</p>']
    if silhouette_gallery():
        page.insert(3,'<p><a href="graphs/silhouettes.html"><strong>New: all interaction graphs with human/object silhouettes and hover labels</strong></a></p>')
        page.append('<p>Silhouette views use the actual cached human mesh and original object surface. The robot row uses the human surface as a translucent reference, not robot geometry. Thick dark/colored anatomy lines are display guides, not extra graph edges or constraints.</p>')
    if summary.get('interpretation'):
        page.insert(4,'<div class="note">'+''.join('<p>'+html.escape(s)+'</p>' for s in summary['interpretation'])+'</div>')
    for seq in COHORT:
        page.append('<h2>'+html.escape(seq)+'</h2>')
        failed=[e for e in summary['failures'] if e['sequence']==seq]
        for failure in failed:
            retry=next((r for r in summary['unchanged_retries'] if r['sequence']==seq and r['variant']==failure['variant']),None)
            detail=f' Identical retry: {retry["state"]}, zero-based frame {retry.get("frame_zero_based","unknown")}.' if retry else ''
            page.append('<p class="note">'+html.escape(LABELS[failure['variant']]+' failed to complete (Clarabel InsufficientProgress).'+detail+' It is excluded from trajectory averages and has no completed video panel.')+'</p>')
        selected=[r for r in rows if r['sequence']==seq]
        page.append(table(selected,[LABELS[r['variant']] for r in selected]))
        if (OUT/'renders'/f'{seq}.mp4').exists():page.append(f'<video controls preload="metadata" src="renders/{seq}.mp4" poster="renders/{seq}.png"></video><a href="renders/{seq}.mp4">Download MP4</a> · <a href="renders/{seq}_hands.png">Hand close-up PNG</a>')
        if (OUT/'renders'/f'{seq}_posture.png').exists():page.append(f'<p><a href="renders/{seq}_posture.png">Arm posture at the largest joint-angle change during contact</a></p>')
        gdir=OUT/'graphs'/seq
        if (gdir/'graphs.json').exists():
            for fr in read(gdir/'graphs.json')['frames']:
                t=fr['frame'];stem=f'graphs/{seq}/frame_{t:03d}'
                if (OUT/f'{stem}_silhouettes.png').exists():
                    page.append(f'<details><summary>Interaction graphs over human/object silhouettes, frame {t}</summary>'
                        f'<p><a href="{stem}_silhouettes_hover.html">Hover labels: body + hands</a> · '
                        f'<a href="{stem}_hands_silhouettes_hover.html">Hover labels: hand close-ups</a></p>'
                        f'<img loading="lazy" src="{stem}_silhouettes.png">'
                        f'<a href="{stem}_hands_silhouettes.png">Hand close-up PNG</a> · '
                        f'<a href="{stem}_weights.png">Neighbour-weight matrices</a> · '
                        f'<a href="{stem}_graph.npz">Numeric graph data</a></details>')
                else:
                    page.append(f'<details><summary>Actual interaction graphs, frame {t}</summary><img src="{stem}.png"><a href="{stem}_hands.png">Hand graph close-up</a> · <a href="{stem}_weights.png">Neighbour-weight matrices</a> · <a href="{stem}_graph.npz">Numeric graph data</a></details>')
            page.append(f'<details><summary>Local arm influence of each objective</summary><img src="graphs/{seq}/arm_gradient.png"><a href="graphs/{seq}/graphs.json">Graph statistics and gradient values</a></details>')
        for r in selected:
            v=r['variant'];prefix=f'variants/{v}/clips/{seq}'
            page.append(f'<p>{html.escape(LABELS[v])}: <a href="{prefix}/checks.json">unchanged settings/input checks</a> · <a href="{prefix}/comfort.json">comfort definitions/results</a> · <a href="{prefix}/collision_audit.json">collision geometry audit</a></p>')
    (OUT/'index.html').write_text('\n'.join(page))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--visuals-only',action='store_true',help='Update HTML only; preserve all metric files.')
    main(parser.parse_args().visuals_only)
