"""Local comparison of surface-only contact and signed material recovery."""
import csv
import html
from config import *


def main():
    rows = []
    for seq in COHORT:
        for variant in REMOVALS:
            dest = destination(seq, variant)
            if not (dest / 'diagnostics.json').exists():
                continue
            d = read(dest / 'diagnostics.json'); m = read(dest / 'metrics.json')['v2']
            run = read(dest / 'run.json')
            rows.append(dict(sequence=seq, variant=variant, frames=d['frames'],
                correct_side_pct=d['correct_side_contact_pct'], patch_gap_mm=d['physical_patch_gap_mean_mm'],
                gap_target_mae_mm=d['surface_gap_target_mae_mm'],
                intended_patch_recall_pct=d['intended_patch_touch_recall_skin_contact_pct'],
                physical_f1_pct=d['physical_touch_vs_original_source']['f1_pct'],
                strict_physical_f1_pct=d['strict_physical_touch_vs_original_source']['f1_pct'],
                object_pen_max_mm=d['object_max_penetration_mm'],
                object_mean_frame_max_pen_mm=d['object_mean_frame_max_penetration_mm'],
                object_pen_above1mm_pct=d['object_penetration_above_1mm_pct'],
                self_pen_max_mm=d['self_max_penetration_mm'],
                arm_near_limit_pct=d['arm_contact_near_limit5deg_pct'],
                arm_accel_rad_s2=d['arm_acceleration_rms_rad_s2'],
                contact_drift_cm_s=d['relative_surface_witness_drift_cm_s'],
                palm_error_deg=m.get('palm_normal_error_deg'), solve_s=run['joint_refinement_seconds'],
                solve_reused=variant == 'surface'))
    if rows:
        with (OUT / 'per_sequence.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write(OUT / 'summary.json', dict(rows=rows, completed=len(rows), total=4,
          caution='Two development clips; kinematic metrics only. Baseline solve times reused.'))
    doc = ['<!doctype html><meta charset="utf-8"><title>V2 signed contact recovery</title>',
        '<style>body{font:16px system-ui;margin:30px;max-width:1800px}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ccc}video,img{max-width:100%}pre{white-space:pre-wrap}.table{overflow:auto}</style>',
        '<h1>V2 step 3: signed material recovery</h1>',
        '<p>Human demonstration | surface-only baseline | surface-only plus signed recovery.</p>',
        '<p>New objective: E = E_surface_only + 256 Σ min(d_material, 0)², on the unchanged source-contact fingertips. Regional unsigned gap and direction remain unchanged; legacy point attraction remains disabled.</p>',
        '<p>The open source patch defines location, not an inside/outside. Signed recovery uses the existing solid convex collision pieces. This is an added recovery residual, not a sign attached to the open-patch gap, not a new voxel SDF, and not the full OTRetarget method.</p>',
        '<p><a href="protocol.json">Exact protocol</a> · <a href="status.json">Status</a> · <a href="per_sequence.csv">CSV</a> · <a href="contact_threshold_sensitivity.csv">Threshold sensitivity</a> · <a href="preflight.json">Sign, overlap and gradient checks</a> · <a href="validation.json">Validation</a></p>',
        '<p>Evaluation definitions unchanged. Contact F1 uses ≤2 mm separation and ≤1 mm penetration against the original source labels. A separate strict column allows ≤10 µm penetration. Intended-patch recall also requires the demonstrated region. See the full sensitivity grid.</p>',
        '<p>Baseline artifacts are reused byte-for-byte. New recovery acts only on active distal fingers; unchanged collision constraints cover all registered arm/hand links. Per-link audits identify residual penetration elsewhere. No physics grasp test has been run.</p>']
    doc.append('<p>Numerical policy recorded for this experiment: '+html.escape(read(OUT / 'protocol.json')['numerical_recovery'])+'</p>')
    for seq in COHORT:
        doc.append('<h2>' + html.escape(seq) + '</h2>')
        subset = [r for r in rows if r['sequence'] == seq]
        if subset:
            keys = [k for k in subset[0] if k not in ('sequence', 'frames')]
            doc.append('<div class="table"><table><tr>' + ''.join('<th>' + k + '</th>' for k in keys) + '</tr>')
            for r in subset:
                doc.append('<tr>' + ''.join('<td>' + html.escape(f'{r[k]:.3f}' if isinstance(r[k], float) else str(r[k])) + '</td>' for k in keys) + '</tr>')
            doc.append('</table></div>')
        audit = OUT / 'link_audit' / f'{seq}.json'
        if audit.exists():
            doc.append(f'<p><a href="link_audit/{seq}.csv">Object penetration by robot link (CSV)</a></p>')
            doc.append('<details><summary>Links with largest residual penetration</summary><pre>' + html.escape(__import__('json').dumps(read(audit), indent=2)) + '</pre></details>')
        path = OUT / 'renders' / f'{seq}.mp4'
        if path.exists():
            doc.append(f'<p><a href="renders/{path.name}">Download MP4</a></p><video controls preload="metadata" src="renders/{path.name}"></video>')
        for suffix in ('', '_hands', '_posture'):
            path = OUT / 'renders' / f'{seq}{suffix}.png'
            if path.exists():
                doc.append(f'<p><a href="renders/{path.name}"><img src="renders/{path.name}"></a></p>')
    (OUT / 'index.html').write_text('\n'.join(doc))


if __name__ == '__main__':
    main()
