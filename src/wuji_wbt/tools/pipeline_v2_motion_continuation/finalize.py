"""Validate exports and add a concise interpretation without changing metrics."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
from collections import Counter
from PIL import Image
import imageio_ffmpeg
from config import *

def main():
    status=read(OUT/'status.json');assert status['state']=='complete'
    assert read(OUT/'validation.json')['passed']
    protocol=read(OUT/'protocol.json')
    for group in ('source_sha256','input_sha256','target_sha256'):
        for path,expected in protocol[group].items():assert sha(path)==expected,(group,path)
    records=[];videos=[]
    for seq in COHORT:
        for variant in VARIANTS:
            dest=destination(seq,variant)
            iterations=read(dest/'iterations.json');steps=[s for frame in iterations for s in frame['steps']]
            assert all(s['numerical_objective_scale']==1. and s['maximum_cone_constraint_violation']<=1e-6 for s in steps)
            assert sha(dest/'final_v2.npz')==read(dest/'run.json')['final_sha256']
            records.append(dict(sequence=seq,variant=variant,qp_count=len(steps),
                statuses=dict(Counter(s['status'] for s in steps)),
                max_cone_constraint_violation=max(s['maximum_cone_constraint_violation'] for s in steps),
                certified_fallbacks=sum(s['accepted_feasible_fallback'] for s in steps)))
        T=read(destination(seq,'baseline')/'run.json')['frames']
        for suffix in ('','_hands','_thumb'):
            path=OUT/'renders'/f'{seq}{suffix}.mp4'
            reader=imageio_ffmpeg.read_frames(str(path),input_params=['-threads','1'],output_params=['-threads','1'])
            meta=next(reader);count=sum(1 for _ in reader)
            assert count==(T+1)//2 and meta['size']==(2400,540) and abs(meta['fps']-15)<1e-6
            videos.append(dict(path=str(path),frames=count,fps=meta['fps'],size=meta['size'],sha256=sha(path)))
        for suffix in ('','_hands','_thumb','_posture'):
            with Image.open(OUT/'renders'/f'{seq}{suffix}.png') as im:im.verify()
    write(OUT/'final_validation.json',dict(passed=True,qp_records=records,videos=videos,
        reporting_scripts_sha256={str(p):sha(p) for p in (CODE/'plots.py',CODE/'finalize.py')},
        note='Endpoint joint/rate/body/source-mask checks and fixed metric definitions verified in validation.json. Collision guards are not an absolute zero-penetration guarantee.'))
    rows=read(OUT/'summary.json')['rows']
    lines=['Geometry side, signed recovery OFF: initialization/continuation comparison',
           'Date: 2026-10-20. Two complete HUMOTO development clips, four conditions each.',
           '', 'Object size and trajectory',
           'Both visible/collision geometry and floor-anchored translations use scale 0.8188585607940446.',
           'p_scaled = s*(p_original - [0,0,z_min]); object rotations unchanged.',
           'Robot geometry is not scaled. Body/object trajectories are identical in every output.',
           'Signed-recovery objective OFF; signed collision checking ON; thumb/index multiplier 1.',
           '', 'Finding',
           'Motion-compensated initialization alone is the best tested trade-off for jitter, large penetration and refinement runtime.',
           'Substeps and combined reduce object penetration similarly, but cost much more and produce an organizer self-collision of 9.97 mm at 0.633 s (left upper arm vs hip geometry).',
           'These are experimental outputs. The previously selected geometry-side baseline has not been overwritten or automatically replaced.',
           '', 'Full-clip metrics (same definitions as baseline):',
           'Sequence | condition | F1% (1mm penetration gate) | max object mm | max self mm | patch gap mm | finger jerk RMS rad/s^3 | refinement seconds']
    for r in rows:
        lines.append(f"{r['sequence']} | {r['variant']} | {r['physical_f1_pct']:.3f} | {r['object_max_pen_mm']:.3f} | {r['self_max_pen_mm']:.3f} | {r['patch_gap_mm']:.3f} | {r['finger_jerk_rad_s3']:.3f} | {r['solve_seconds']:.3f}")
    lines += ['', 'Contact-score caveat',
        'The primary F1 requires gap <=2 mm and penetration <=1 mm. The unchanged optimizer allows 1 mm penetration plus penalized slack.',
        'Many improved trajectories remain approximately 1.004-1.04 mm inside the surface, so primary F1 decreases despite smaller maximum depths and patch gaps.',
        'Do not replace that primary score silently. At the predeclared 1.5 mm sensitivity gate:',
        'Organizer baseline 97.794%, init 98.909%, substeps 98.933%, combined 98.933%.',
        'Rack baseline 88.662%, all three new conditions 100%.',
        'Strict F1, other thresholds, intended-patch recall, contact geometry and all original metrics remain in the CSV/JSON files.',
        'This does not establish force-closure or successful physical grasping. Early organizer finger transients remain; not all whole-clip motion jerk is removed.',
        '', 'Known event windows',
        'Organizer frames 105..120: max finger change 11.459->1.029 deg/frame and max object depth 9.510->1.033 mm with init alone.',
        'Rack frames 73..90: max finger change 5.349->0.627 deg/frame and max object depth 1.560->1.056 mm with init alone.',
        '', 'Implementation and limitations',
        'Init: object-rigid-motion palm target, bounded arm IK predictor, existing fingers retained for initialization, then unchanged coupled arm/finger objective.',
        'Substeps: adaptive body/object interpolation, unchanged endpoint targets, per-substep rate budget proportional to time fraction; same 0.2 rad/frame total.',
        'Signed collision guards sample intermediate configurations and limit incremental penetration growth. They are not continuous collision detection or an absolute penetration cap; repeated small allowed increases can accumulate.',
        'Organizer substeps-only hit the resource cap once; all such events are recorded. Zero unresolved guards does not imply zero penetration.',
        'Same objective weights, source masks, evaluation thresholds and 30 Hz outputs. More inner solves are allowed for substeps; the extra runtime is reported.',
        'Runtime excludes reused Omni body generation and quick-finger preprocessing. Shared-workstation wall timings, one CPU thread per job.',
        'Baseline/init ran with 2 workers; remaining jobs ran with 4 after a documented resource-only restart. Interrupted incomplete logs are archived; completed results were preserved.',
        f"Batch elapsed (including interrupted-attempt time, evaluations and renders): {status['elapsed_seconds']:.1f} seconds.",
        '', 'Artifacts: index.html (five-panel videos), per_sequence.csv, changes_vs_baseline.json, plots/, object_scaling_audit.json, final_validation.json.']
    (OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    path=OUT/'index.html';doc=path.read_text()
    if 'id="final-interpretation"' not in doc:
        doc += '<section id="final-interpretation"><h2>Interpretation</h2><p>Initialization alone is the best tested jitter/penetration/runtime trade-off. Substeps add little contact benefit and cause a 9.97 mm organizer self-collision. Primary F1 falls because contacts cluster just beyond the unchanged 1 mm penetration gate; the full predeclared sensitivity table is retained.</p><p><a href="RESULTS.txt">Detailed interpretation and limitations</a> · <a href="object_scaling_audit.json">Object scaling audit</a> · <a href="final_validation.json">Export and numerical validation</a></p>'
        for seq in COHORT:doc+=f'<p><a href="plots/{seq}.png">Full-clip and event plot: {seq}</a></p><img src="plots/{seq}.png">'
        path.write_text(doc+'</section>')
    print('Validated eight trajectories, six five-panel MP4s and all PNGs.',flush=True)

if __name__=='__main__':main()
