"""Generate a separate comparison report without modifying earlier results."""
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
            d = read(dest / 'diagnostics.json')
            m = read(dest / 'metrics.json')['v2']
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
                solve_reused=variant == 'both'))
    if rows:
        with (OUT / 'per_sequence.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write(OUT / 'summary.json', dict(rows=rows, completed=len(rows), total=4,
          caution='Two development clips; kinematic metrics only. Baseline runtime reused, not a fresh timing benchmark.'))
    doc = ['<!doctype html><meta charset="utf-8"><title>V2 contact target replacement</title>',
        '<style>body{font:16px system-ui;margin:30px;max-width:1800px}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ccc}video,img{max-width:100%}pre{white-space:pre-wrap}.table{overflow:auto}</style>',
        '<h1>V2 step 2: contact target replacement</h1>',
        '<p>Human demonstration | previous combined version | combined without legacy point attraction.</p>',
        '<p>Only contact_weight changes: 100 → 0. Surface gap (256), direction (2500), source patches, all other objectives, collision settings, initialization and body/object trajectories are unchanged. Human fingertip tracking remains enabled.</p>',
        '<p><a href="protocol.json">Protocol</a> · <a href="status.json">Status</a> · <a href="per_sequence.csv">CSV</a> · <a href="contact_threshold_sensitivity.csv">Threshold sensitivity</a> · <a href="preflight.json">Exact objective checks</a> · <a href="validation.json">Output validation</a></p>',
        '<p>Existing evaluation definitions are unchanged. The table reports the preceding experiment’s supplemental geometry diagnostics. Physical F1 uses a 2-mm separation gate and 1-mm penetration gate; inspect the complete sensitivity grid because the unchanged solver permits 1 mm penetration plus penalized slack. Unsigned gap can be zero at intersection.</p>',
        '<p>Previous combined trajectories and metrics are reused after hash verification. Their solve times are historical. No physical grasp or load-bearing test has been run.</p>']
    for seq in COHORT:
        doc.append('<h2>' + html.escape(seq) + '</h2>')
        subset = [r for r in rows if r['sequence'] == seq]
        if subset:
            keys = [k for k in subset[0] if k not in ('sequence', 'frames')]
            doc.append('<div class="table"><table><tr>' + ''.join('<th>' + k + '</th>' for k in keys) + '</tr>')
            for r in subset:
                doc.append('<tr>' + ''.join('<td>' + html.escape(f'{r[k]:.3f}' if isinstance(r[k], float) else str(r[k])) + '</td>' for k in keys) + '</tr>')
            doc.append('</table></div>')
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
