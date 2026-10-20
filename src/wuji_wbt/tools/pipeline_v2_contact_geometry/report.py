"""Small local report; generated HTML, media and data remain outside code worktree."""
import csv
import html
from config import *

def main():
    rows=[]
    for seq in COHORT:
        for variant in REMOVALS:
            dest=destination(seq,variant)
            if not (dest/'diagnostics.json').exists():continue
            d=read(dest/'diagnostics.json');m=read(dest/'metrics.json')['v2'];run=read(dest/'run.json')
            rows.append(dict(sequence=seq,variant=variant,frames=d['frames'],
                correct_side_pct=d['correct_side_contact_pct'],patch_gap_mm=d['physical_patch_gap_mean_mm'],
                gap_target_mae_mm=d['surface_gap_target_mae_mm'],
                intended_patch_recall_pct=d['intended_patch_touch_recall_skin_contact_pct'],
                physical_f1_pct=d['physical_touch_vs_original_source']['f1_pct'],
                strict_physical_f1_pct=d['strict_physical_touch_vs_original_source']['f1_pct'],
                object_pen_max_mm=d['object_max_penetration_mm'],object_pen_above1mm_pct=d['object_penetration_above_1mm_pct'],
                self_pen_max_mm=d['self_max_penetration_mm'],
                arm_near_limit_pct=d['arm_contact_near_limit5deg_pct'],
                arm_accel_rad_s2=d['arm_acceleration_rms_rad_s2'],
                contact_drift_cm_s=d['relative_surface_witness_drift_cm_s'],
                palm_error_deg=m.get('palm_normal_error_deg'),solve_s=run['joint_refinement_seconds']))
    if rows:
        with (OUT/'per_sequence.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(OUT/'summary.json',dict(rows=rows,completed=len(rows),total=8,
        caution='Two development clips. Matched control uses Coal. These are kinematic metrics; no load-bearing grasp test.'))
    doc=['<!doctype html><meta charset="utf-8"><title>V2 contact geometry</title>',
         '<style>body{font:16px system-ui;margin:30px;max-width:1800px}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ccc}video,img{max-width:100%}pre{white-space:pre-wrap}</style>',
         '<h1>V2 contact geometry: organizer and clothes rack</h1>',
         '<p>Human | V2 control (Coal) | direction | surface gap | both. No pad-facing term. All original V2 objectives retained.</p>',
         '<p><a href="protocol.json">Exact formulas, provenance, settings and hashes</a> · <a href="status.json">Status</a> · <a href="preflight.json">Jacobian checks</a> · <a href="per_sequence.csv">CSV</a> · <a href="validation.json">Output validation</a></p>',
         '<p>Matched V2 control is rerun with Coal signed collision queries. It is not the earlier native-MuJoCo trajectory. Original metric definitions remain in each clip’s metrics.json; the table below contains separately defined geometry diagnostics.</p>']
    if (OUT/'contact_threshold_sensitivity.csv').exists():
        doc.append('<p><strong>Contact threshold sensitivity:</strong> The unchanged solver allows 1 mm penetration plus penalized slack. The table’s 1-mm-gated F1 is sensitive at that boundary. See the complete <a href="contact_threshold_sensitivity.csv">post-hoc threshold grid</a> and <a href="contact_threshold_sensitivity.json">definitions</a>. These supplement, and do not replace, the primary scores. Gap is unsigned and can be zero on intersection; inspect penetration alongside it.</p>')
    for seq in COHORT:
        doc.append('<h2>'+html.escape(seq)+'</h2>')
        subset=[r for r in rows if r['sequence']==seq]
        if subset:
            keys=[k for k in subset[0] if k not in ('sequence','frames')]
            doc.append('<table><tr>'+''.join('<th>'+k+'</th>' for k in keys)+'</tr>')
            for r in subset:
                doc.append('<tr>'+''.join('<td>'+html.escape(f'{r[k]:.3f}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in keys)+'</tr>')
            doc.append('</table>')
        audit=OUT/'targets'/f'{seq}.json'
        if audit.exists():doc.append('<details><summary>Source contact audit</summary><pre>'+html.escape(json.dumps(read(audit),indent=2))+'</pre></details>')
        for suffix in ('','_hands','_posture'):
            path=OUT/'renders'/f'{seq}{suffix}.png'
            if path.exists():doc.append(f'<a href="renders/{path.name}"><img src="renders/{path.name}"></a>')
        path=OUT/'renders'/f'{seq}.mp4'
        if path.exists():doc.append(f'<video controls preload="metadata" src="renders/{path.name}"></video>')
        path=OUT/'targets'/f'{seq}_source_contacts.png'
        if path.exists():doc.append(f'<details><summary>Source hand, object patches and directions</summary><img src="targets/{path.name}"></details>')
    protocol=read(OUT/'protocol.json')
    doc.append('<h2>Added terms and differences from papers</h2>')
    for key in ('direction','gap','dexweave','weights','collision_control','evaluation','limitations'):
        doc.append('<h3>'+key+'</h3><pre>'+html.escape(json.dumps(protocol[key],indent=2))+'</pre>')
    (OUT/'index.html').write_text('\n'.join(doc))

if __name__=='__main__':main()
