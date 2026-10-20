"""Collect all declared trials and render the final side-constraint candidate."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
import csv
import html
import subprocess
import time
from config import *


def main():
    assert read(OUT / 'status.json')['state'] == 'complete'
    seq = COHORT[0]
    first = ROOT / 'holosoma/eval_out/ref_candidates/pipeline_v2_thumb_contact_20261019'
    trials = [(first, 'baseline', 'Current signed-contact V2'),
              (first, 'priority', 'Thumb/index contact costs ×4'),
              (first, 'support', 'Physical fingertip side'),
              (BASELINE, 'touch', 'Physical side + zero thumb gap'),
              (BASELINE, 'combined', 'Physical side + thumb/index ×4'),
              (OUT, 'hard', 'Combined + right-thumb side inequality')]
    rows = []
    for folder, variant, label in trials:
        audit = read(folder / 'audit' / f'{variant}.json')
        dest = folder / 'variants' / variant / 'clips' / seq
        d = read(dest / 'diagnostics.json'); run = read(dest / 'run.json')
        rows.append(dict(method=label, variant=variant, result_directory=str(dest),
            thumb_inside_pct=audit['right_thumb_correct_side_pct'],
            other_four_outside_pct=audit['right_other_fingers_correct_side_pct'],
            thumb_intended_contact_pct=audit['right_thumb_intended_contact_pct'],
            thumb_gap_mm=audit['right_thumb_patch_gap_mm'],
            thumb_max_pen_mm=audit['right_thumb_max_material_pen_mm'],
            original_f1_pct=d['physical_touch_vs_original_source']['f1_pct'],
            strict_f1_pct=d['strict_physical_touch_vs_original_source']['f1_pct'],
            object_max_pen_mm=d['object_max_penetration_mm'],
            object_mean_frame_max_pen_mm=d['object_mean_frame_max_penetration_mm'],
            self_max_pen_mm=d['self_max_penetration_mm'],
            arm_accel_rad_s2=d['arm_acceleration_rms_rad_s2'],
            arm_near_limit_pct=d['arm_contact_near_limit5deg_pct'],
            solve_seconds=run['joint_refinement_seconds']))
    validation = read(OUT / 'validation.json')
    write(OUT / 'summary.json', dict(sequence=seq, rows=rows, validation=validation,
        interpretation='Every metric uses the original source labels and evaluators. Thumb-specific side/contact statistics are supplemental. The zero-gap trial changes an optimization target explicitly; it does not alter evaluation labels.'))
    with (OUT / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    start = time.perf_counter()
    with (OUT / 'render.log').open('w') as stream:
        subprocess.run([PY, str(CODE / 'render.py'), seq], cwd=WORK, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    sources = {str(CODE / f): sha(CODE / f) for f in ('finish.py', 'render.py')}
    artifacts = {str(p.relative_to(OUT)): sha(p) for p in (OUT / 'cavity_renders').glob('*') if p.suffix in ('.mp4', '.png', '.json')}
    write(OUT / 'render_validation.json', dict(source_sha256=sources, artifacts_sha256=artifacts,
        render_seconds=time.perf_counter()-start, same_frames_cameras_objects=True, cpu_only=True))
    doc = ['<!doctype html><meta charset="utf-8"><title>Organizer thumb contact diagnosis</title>',
        '<style>body{font:16px system-ui;margin:30px}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #aaa}video,img{max-width:100%}pre{white-space:pre-wrap}</style>',
        '<h1>Organizer: keep the right thumb inside and four fingers outside</h1>',
        '<p>The source target already selects the correct inner face. At frame 120 the baseline right-thumb reference is 11.73 mm on the wrong side. The original side loss only measures a reference point; it does not enforce which side the entire distal geometry occupies. The local source-skin gap target is 3.39 mm, so increasing that weight does not necessarily request touch. Numerical contact derivatives were checked.</p>',
        '<p>Geometry-side residual: min(0, min over distal-tip vertices of n·(v−anchor) + 1 mm). The normal and finite surface patch come from the original human demonstration. This uses the existing collision convex, not a newly chosen pad point.</p>',
        '<p>Combined variant: the geometry-side replacement plus ×4 direction/gap/signed-recovery costs for thumb and index on both hands. Original human fingertip tracking and interaction-mesh weights stay fixed. The final trial additionally constrains the right-thumb side in each QP during source contact, with no slack on that inequality. Its final nonlinear clearance is checked below.</p>',
        '<p><a href="comparison.csv">All trial results (CSV)</a> · <a href="protocol.json">Constraint protocol</a> · <a href="validation.json">Nonlinear validation</a> · <a href="summary.json">All measurements</a></p>',
        '<p>Side percentages use existing source-contact frames and a 1 mm tolerance. Intended contact additionally requires ≤2 mm gap to the intended patch and ≤1 mm penetration. Original F1 uses the original source labels and all object surfaces; it does not establish the correct wall-side placement.</p>',
        '<div style="overflow:auto"><table><tr>']
    columns = [k for k in rows[0] if k not in ('variant', 'result_directory')]
    doc += [''.join('<th>'+k+'</th>' for k in columns)+'</tr>']
    for row in rows:
        doc.append('<tr>'+''.join('<td>'+html.escape(f'{row[k]:.3f}' if isinstance(row[k], float) else str(row[k]))+'</td>' for k in columns)+'</tr>')
    doc.append('</table></div><p>Final nonlinear side check: '+str(validation['nonlinear_side_passed'])+
        '; maximum excess beyond the 1 mm plane tolerance: '+f"{validation['nonlinear_side_violation_m']*1000:.6f}"+' mm.</p>')
    for suffix, title in (('_thumb', 'Look inside the organizer'), ('', 'Whole-body comparison')):
        doc.append(f'<h2>{title}</h2><p>Human | original signed contact | geometry side + contact priority | additional side inequality</p><p><a href="cavity_renders/{seq}{suffix}.mp4">Download MP4</a></p><video controls preload="metadata" src="cavity_renders/{seq}{suffix}.mp4"></video>')
    doc.append(f'<p><img src="cavity_renders/{seq}_thumb.png"></p>')
    doc.append('<p>These are kinematic references on one development clip. Correct side and small surface gap do not establish pad-facing quality, force closure or physical lifting. Other links may still penetrate; the full-object maximum is reported separately.</p>')
    (OUT / 'index.html').write_text('\n'.join(doc))


if __name__ == '__main__': main()
