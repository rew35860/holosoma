"""Report a focused per-digit audit alongside unchanged whole-clip metrics."""
import csv
import html
from config import *


def main():
    seq = COHORT[0]; rows = []
    for variant in ('baseline', 'priority', 'support'):
        a = read(OUT / 'audit' / f'{variant}.json')
        d = read(destination(seq, variant) / 'diagnostics.json')
        run = read(destination(seq, variant) / 'run.json')
        row = {k: v for k, v in a.items() if k not in ('definitions', 'sequence', 'frames')}
        row.update(f1_pct=d['physical_touch_vs_original_source']['f1_pct'],
            strict_f1_pct=d['strict_physical_touch_vs_original_source']['f1_pct'],
            object_max_pen_mm=d['object_max_penetration_mm'],
            object_mean_frame_max_pen_mm=d['object_mean_frame_max_penetration_mm'],
            self_max_pen_mm=d['self_max_penetration_mm'],
            arm_accel_rad_s2=d['arm_acceleration_rms_rad_s2'],
            arm_near_limit_pct=d['arm_contact_near_limit5deg_pct'],
            solve_s=run['joint_refinement_seconds'])
        rows.append(row)
    write(OUT / 'summary.json', dict(sequence=seq, rows=rows, diagnosis=read(OUT / 'diagnosis.json')))
    with (OUT / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    doc = ['<!doctype html><meta charset="utf-8"><title>Organizer thumb contact</title>',
        '<style>body{font:16px system-ui;margin:30px}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #bbb;padding:7px}video,img{max-width:100%}pre{white-space:pre-wrap}</style>',
        '<h1>Organizer: right thumb inside, four fingers outside</h1>',
        '<p>Matched numerical settings. Human demonstration | current signed contact | thumb/index contact costs ×4 | physical fingertip side constraint (soft). Body and object unchanged.</p>',
        '<p>The baseline source already selects the inner face for the right thumb and the outer face for the other four digits. The experiments test stronger contact weights versus accounting for the actual extent of the fingertip in the direction term.</p>',
        '<p>Original F1: ≤2 mm separation and ≤1 mm penetration against unchanged source labels. It does not establish correct wall-side placement. Per-digit side and intended-patch measurements below supplement it.</p>',
        '<p><a href="comparison.csv">CSV</a> · <a href="protocol.json">Protocol</a> · <a href="diagnosis.json">Diagnosis</a> · <a href="status.json">Status</a></p>']
    columns = list(rows[0]); doc.append('<div style="overflow:auto"><table><tr>'+''.join('<th>'+c+'</th>' for c in columns)+'</tr>')
    for row in rows:
        doc.append('<tr>'+''.join('<td>'+html.escape(f'{row[c]:.3f}' if isinstance(row[c], float) else str(row[c]))+'</td>' for c in columns)+'</tr>')
    doc.append('</table></div>')
    for suffix in ('', '_thumb'):
        doc.append(f'<h2>{"Right-hand detail" if suffix else "Whole sequence"}</h2><video controls preload="metadata" src="renders/{seq}{suffix}.mp4"></video>')
    for suffix in ('_hands', '_thumb'):
        doc.append(f'<p><img src="renders/{seq}{suffix}.png"></p>')
    doc.append('<p>This is a kinematic contact test. No force-closure or physical lifting validation has been performed.</p>')
    (OUT / 'index.html').write_text('\n'.join(doc))


if __name__ == '__main__': main()
