"""Prespecify the signed-recovery ablation and freeze the completed baseline."""
import shutil
from config import *


def main():
    assert not OUT.exists(), 'Never overwrite previous results'
    previous = read(BASE_OUT / 'protocol.json')
    assert read(BASE_OUT / 'validation.json')['passed']
    assert read(BASE_OUT / 'status.json')['state'] == 'complete'
    for group in ('input_sha256', 'source_sha256', 'target_sha256'):
        for path, expected in previous[group].items():
            assert sha(path) == expected, path
    OUT.mkdir(parents=True)
    shutil.copytree(BASE_OUT / 'targets', OUT / 'targets')
    reused = {}
    for seq in COHORT:
        source = BASE_OUT / 'variants/surface/clips' / seq
        dest = destination(seq, 'surface')
        shutil.copytree(source, dest)
        reused[seq] = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    sources = set(Path(p) for p in previous['source_sha256']) | set(CODE.glob('*.py'))
    sources.add(RET / 'retarget_pipeline/clarabel_step.py')
    protocol = dict(version='v2-signed-contact-step3-1', experiment_date='2026-10-19',
        branch='feature/wuji-pipeline-v2', sequences=COHORT, predeclared_before_optimization=True,
        baseline_directory=str(BASE_OUT),
        variants={'surface': dict(label=LABELS['surface'], reused=True),
                  'signed': dict(label=LABELS['signed'], reused=False)},
        objective_change='E_signed = E_surface_only + 256 sum_{source-contact fingertips} min(d_material, 0)^2.',
        signed_distance='d_material is the minimum signed convex-pair separation over existing object material pieces. Positive outside, negative on overlap. This is not an exact Euclidean SDF of the nonconvex union.',
        rationale='Open finite source patches do not enclose a material volume. Preserve regional unsigned attraction, and add a signed recovery residual instead of assigning an arbitrary sign to an open-patch distance. No replacement with nearest-object-only attraction.',
        numerical_recovery='Unscaled QP with gap abs/rel 1e-6 and feasibility 1e-7; reduced defaults unchanged. Verify original-unit cone feasibility within 1e-6 for all steps. For InsufficientProgress/MaxIterations require an independent dual-bound gap certificate as well. No post-solve clipping. Certified fallback events are logged. Reused historical outputs retain their original numerical policy.',
        interpretation='The change is an added signed material recovery residual, not a literal signed version of the old shape-to-patch gap and not the full OTRetarget method.',
        recovery=dict(weight=SIGNED_WEIGHT, target=0., activation='Unchanged source-contact fingertip mask and d_material < 0',
                      points='Whole existing distal fingertip collision convex, one minimum separation per active fingertip'),
        unchanged=['Legacy contact attraction remains disabled', 'Regional surface gap weight 256',
                   'Interaction direction weight 2500 and source anchors', 'Human fingertip tracking 25',
                   'Interaction mesh 500, fingertip row multiplier 10', 'All posture, bone and temporal terms',
                   'Signed collision constraints, pairs, 1-mm tolerance and 1e5 slack weight',
                   '54 active joints, quick-finger initialization, body and object trajectories'],
        excluded=['New pad-facing objective', 'New finger priorities', 'OT surface correspondences',
                  'New signed voxel grid', 'Finger-link collision geometry changes', 'Physics validation'],
        extra=EXTRA,
        evaluation='Same frozen historical evaluator and supplemental contact-geometry metrics, masks, thresholds and aggregation. Evaluator always measures the original unsigned intended-patch gap.',
        supplemental_threshold_grid_mm=[.01, .1, .5, 1., 1.5, 2., 5.],
        additional_diagnostics='Per-link registered arm/hand-object signed penetration, measured identically for baseline and new output. No evaluation threshold is replaced.',
        resources=dict(workers=2, cpu_quota_percent=200, nice=19, memory_max_gib=5, memory_swap_max=0, gpu=False),
        limitations=['Two development clips; no force or load-bearing grasp test.',
                     'The signed recovery objective covers active distal fingers only; collision constraints still cover all registered pairs.',
                     'Convex decomposition approximates material, and minima across pieces are not exact nonconvex penetration depths.',
                     'Contact region, reference-point direction and unsigned gap limitations remain.',
                     'Baseline timings are historical; only two new solves are run.'],
        input_sha256=previous['input_sha256'], source_sha256={str(p): sha(p) for p in sorted(sources)},
        target_sha256={str(p): sha(p) for p in (OUT / 'targets').glob('*.npz')},
        reused_baseline_sha256=reused)
    write(OUT / 'protocol.json', protocol)
    for source in sources:
        dest = OUT / 'source_snapshot' / source.relative_to(WORK)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    write(OUT / 'status.json', dict(state='prepared', completed=2, total=4,
          new_solves_completed=0, new_solves_total=2))


if __name__ == '__main__':
    main()
