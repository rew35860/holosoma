"""Freeze targets, baseline outputs, definitions and source hashes before solving."""
import shutil
from config import *


def main():
    assert not OUT.exists(), 'Never overwrite an existing experiment'
    prior = read(BASE_OUT / 'protocol.json')
    assert read(BASE_OUT / 'validation.json')['passed']
    assert read(BASE_OUT / 'status.json')['state'] == 'complete'
    for group in ('input_sha256', 'source_sha256', 'target_sha256'):
        for path, expected in prior[group].items():
            assert sha(path) == expected, path
    OUT.mkdir(parents=True)
    shutil.copytree(BASE_OUT / 'targets', OUT / 'targets')
    reused = {}
    for seq in COHORT:
        source = BASE_OUT / 'variants/both/clips' / seq
        dest = destination(seq, 'both')
        shutil.copytree(source, dest)
        reused[seq] = {str(path.relative_to(source)): sha(path)
                       for path in source.rglob('*') if path.is_file()}
    sources = set(Path(p) for p in prior['source_sha256']) | set(CODE.glob('*.py'))
    source_hashes = {str(p): sha(p) for p in sorted(sources)}
    gap = dict(prior['gap'])
    gap['differences'] = ('Same unsigned shape-to-source-patch gap as the preceding experiment. '
                         'Legacy point attraction is retained only in the reused both baseline '
                         'and disabled in surface. Human fingertip tracking remains in both.')
    protocol = dict(
        version='v2-surface-contact-step2-1', experiment_date='2026-10-19',
        sequences=COHORT, branch='feature/wuji-pipeline-v2',
        predeclared_before_optimization=True, baseline_directory=str(BASE_OUT),
        variants={'both': dict(label=LABELS['both'], legacy_contact_weight=100., reused=True),
                  'surface': dict(label=LABELS['surface'], legacy_contact_weight=0., reused=False)},
        objective_change='Only contact_weight changes from 100 to 0. The existing direction and unsigned regional surface-gap residuals are retained with identical targets, masks and weights.',
        interpretation='Surface target only refers to the two contact-attraction representations. Human fingertip tracking and the interaction mesh remain enabled.',
        unchanged=['Human fingertip tracking 25', 'Interaction mesh 500 and tip rows x10',
                   'Bone and palm terms', 'Arm and finger priors', 'Temporal regularization',
                   'Source contact masks and skin patches', 'Signed collision backend, pairs, 1-mm tolerance and 1e5 slack penalty',
                   'Original Omni body/object trajectories', '54 active arm/finger joints and quick-finger initialization'],
        excluded=['SDF replacement', 'New pad-facing or pad-point term', 'Finger priority changes',
                  'New posture terms', 'Body/object reretargeting', 'Physics grasp validation'],
        direction=prior['direction'], gap=gap, extra=EXTRA,
        evaluation='Identical frozen historical evaluator and contact-geometry diagnostics from the preceding experiment. No threshold, denominator, mask or aggregation change.',
        supplemental_threshold_grid_mm=[.01, .1, .5, 1., 1.5, 2., 5.],
        threshold_grid_note='Previously post-hoc; explicitly included before solving this comparison. Primary scores remain unchanged.',
        baseline_reuse='All baseline artifacts copied byte-for-byte after verifying prior input/source/target hashes. Baseline runtime is historical, not a fresh timing benchmark.',
        limitations=['Two development clips, not an independent test set.',
                     'Unsigned surface gap can be zero at intersection; inspect signed penetration too.',
                     'Human fingertip tracking and interaction mesh still contain positional guidance.',
                     'No physical grasp or load-bearing claim.'],
        resources=dict(workers=2, cpu_quota_percent=200, nice=19, memory_max_gib=5, memory_swap_max=0, gpu=False),
        input_sha256=prior['input_sha256'], source_sha256=source_hashes,
        target_sha256={str(p): sha(p) for p in (OUT / 'targets').glob('*.npz')},
        reused_baseline_sha256=reused)
    write(OUT / 'protocol.json', protocol)
    for source in sources:
        dest = OUT / 'source_snapshot' / source.relative_to(WORK)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    write(OUT / 'status.json', dict(state='prepared', completed=2, total=4, new_solves_completed=0, new_solves_total=2))


if __name__ == '__main__':
    main()
