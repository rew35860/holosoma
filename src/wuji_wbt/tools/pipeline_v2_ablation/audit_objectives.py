"""Audit cached target offsets and penetration/contact scores; never runs a solver.

Additional threshold sweeps are diagnostics, not replacements for the frozen metrics.
"""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import config as C


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def f1(source, prediction):
    tp = np.count_nonzero(source & prediction)
    fp = np.count_nonzero(~source & prediction)
    fn = np.count_nonzero(source & ~prediction)
    return float(200 * tp / (2 * tp + fp + fn))


def main():
    summary = C.read(C.OUT/'summary.json')
    shared = summary['shared_successful_sequences']
    thresholds = [0, .5, 1, 1.001, 1.01, 1.05, 1.1, 1.5, 2, 5, 10, 20]
    files = [C.OUT/'summary.json']
    rows = []
    arrays = {}
    for variant in C.REMOVALS:
        for seq in C.COHORT:
            dest = C.OUT/'variants'/variant/'clips'/seq
            if not (dest/'metrics.json').exists():
                continue
            files.extend([dest/'metrics.json', dest/'per_frame_metrics.npz'])
            result = C.read(dest/'metrics.json')
            metrics = result['v2']
            with np.load(dest/'per_frame_metrics.npz') as z:
                depth = z['hand_object_depth_m'].max(1)*1000
                source = z['source_contact']
                pred = z['contact_prediction']
                for limit, key in [(1, 'contact_f1_penetration_gated_pct'),
                                   (5, 'contact_f1_penetration_gated5mm_pct')]:
                    gate = z['hand_object_depth_m'][..., None] <= limit/1000
                    assert np.isclose(f1(source, pred & gate), metrics[key], atol=1e-10)
                assert np.isclose(f1(source, pred), metrics['contact_f1_pct'], atol=1e-10)
                obj = np.maximum(depth/1000, z['body_object_depth'])
                violating = (obj > .02) | (z['ground_depth'] > .01) | (z['self_depth'] > .01)
                assert np.isclose(100*violating.mean(), metrics['combined_pen_duration_pct'])
                hand_violation = depth > 20
                row = dict(variant=variant, sequence=seq, shared_cohort=seq in shared,
                           frames=len(depth), coarse_violating_frames=np.flatnonzero(violating).tolist(),
                           hand_object_gt20mm_frames=np.flatnonzero(hand_violation).tolist())
            arrays[variant, seq] = depth
            row.update({key: metrics[key] for key in (
                'contact_f1_pct', 'contact_f1_penetration_gated_pct',
                'contact_f1_penetration_gated5mm_pct', 'contact_location_gap_mm',
                'fingertip_surface_distance_on_contact_mm', 'combined_pen_duration_pct')})
            row['contact_f1_both_labels_10mm_pct'] = 100*result['contact_counts']['contact_counts']['10mm']['f1']
            row['hand_object_pen_mean_mm'] = float(depth.mean())
            row['hand_object_pen_max_mm'] = float(depth.max())
            for threshold in thresholds:
                row[f'hand_object_gt{threshold:g}mm_pct'] = float(100*np.mean(depth > threshold))
            assert np.isclose(row['hand_object_gt1mm_pct'], metrics['hand_object_pen_duration_gt1mm_pct'])
            rows.append(row)

    aggregates = {}
    for cohort, selected in [('shared_five', shared), ('all_successful', C.COHORT)]:
        aggregates[cohort] = {}
        for variant in C.REMOVALS:
            values = [r for r in rows if r['variant'] == variant and r['sequence'] in selected]
            pooled = np.concatenate([arrays[variant, r['sequence']] for r in values])
            keys = [k for k in values[0] if k.endswith(('_pct', '_mm'))]
            aggregates[cohort][variant] = dict(
                clips=len(values), frames=len(pooled),
                clip_equal_means={k: float(np.mean([r[k] for r in values])) for k in keys},
                pooled_frame_max_depth_quantiles_mm={str(q): float(np.percentile(pooled, q))
                                                     for q in [0, 25, 50, 75, 95, 99, 100]})

    sources = []
    pooled_offsets = {}
    for seq in C.COHORT:
        cache = C.baseline_for(seq)/'evaluation_v1/per_frame'/f'{seq}.npz'
        canonical = C.inputs(seq)['canonical']
        files.extend([cache, canonical])
        with np.load(cache) as z:
            assert np.array_equal(z['source_contact'], z['source_tip_distance_m'] < .03)
            offsets = z['source_tip_distance_m'][z['source_contact']]*1000
        with np.load(canonical) as z:
            sample_count = len(z['object_points_scaled'])
        pooled_offsets[seq] = offsets
        sources.append(dict(sequence=seq, object_sample_count=sample_count,
                            contact_tip_frames=len(offsets), source_to_projected_target_mean_mm=float(offsets.mean()),
                            source_to_projected_target_median_mm=float(np.median(offsets)),
                            source_to_projected_target_p95_mm=float(np.percentile(offsets, 95)),
                            source_to_projected_target_gt1mm_pct=float(100*np.mean(offsets > 1))))
    source_summary = {}
    for name, selected in [('shared_five', shared), ('all_six', C.COHORT)]:
        a = np.concatenate([pooled_offsets[s] for s in selected])
        source_summary[name] = dict(contact_tip_frames=len(a), pooled_mean_mm=float(a.mean()),
                                   pooled_median_mm=float(np.median(a)), pooled_gt1mm_pct=float(100*np.mean(a > 1)))

    solver = C.SOLVER
    evaluator = C.WORK/'src/wuji_wbt/tools/ref_candidates_metrics/evaluate.py'
    files.extend([solver, evaluator, Path(__file__)])
    audit = dict(
        purpose='Formula audit and additional small-depth diagnostics; original evaluation definitions unchanged.',
        optimizer_rerun=False, gpu_used=False, source_files_sha256={str(p): sha(p) for p in dict.fromkeys(files)},
        shared_successful_sequences=shared, aggregates=aggregates, per_clip=rows,
        source_target_offsets=sources, source_target_offset_summary=source_summary,
        definitions={
            'depth': 'Per frame maximum hand-object penetration across both hands, in mm; native collision geometries vs object convex pieces.',
            'aggregation': 'Equal weight to each successful clip. Pooled-frame quantiles explicitly separate. Shared five exclude organizer because three variants failed.',
            'strict_contact': 'Source and predicted fingertip proximity <30mm to original triangle mesh. Invalidate all five predictions of a hand if any geometry of that hand penetrates object by >1mm (or >5mm).',
            '10mm_contact': 'Both source and predicted contact labels use 10mm instead of 30mm; not a penetration gate. Existing cached metric, reported separately.',
            'coarse': 'Object >20mm OR ground/self >10mm; equal-clip average duration. Small-depth sweeps do not replace this metric.',
            'projection_offset': 'Unsigned distance between source human fingertip and its closest surface point, restricted to original 30mm source-contact labels.'})
    C.write(C.OUT/'objective_audit.json', audit)
    with (C.OUT/'objective_audit_per_clip.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    report = [
        'V2 OBJECTIVE AND STRICT-CONTACT AUDIT', '',
        'Cached trajectories only. No optimization rerun, no GPU use, and no evaluation definition changes.',
        'The following formulas describe each hand; sum over left and right hands and frames as applicable.', '',
        'Robot points: 21 per hand: palm origin as wrist proxy; link1/link3/link4 origins as MCP/PIP/DIP;',
        'five fingertip collision-geometry origins as TIP proxies. These are not surface contact-pad samples.', '',
        'Interaction mesh: V=[H;O] in the object frame; H=21 human points, O=fixed cached object samples.',
        'A 3D Delaunay tetrahedral graph defines neighbours. w_ij=exp(-30||V_i-V_j||)/(sum_neighbours exp(-30||V_i-V_k||)+1e-9).',
        'L=I-W. D has sqrt(10) on the five fingertip rows, 1 on every other row (including object rows).',
        'E_IM=(500/Nv)||D(L[P;O]-L[H;O])||_F^2=(500/Nv)||D L_H(P-H)||_F^2.',
        'Object coordinates cancel in the residual, but their source-dependent graph weights and anchoring remain.',
        'Graph and weights rebuild from the human demonstration each frame and stay fixed during its SQP solve.',
        'Contact labels never switch graph edges, lambda_IM, or fingertip row weights.', '',
        'Fingertip tracking: E_tip=25 sum_f ||p_f-h_f||^2 in world coordinates, on all ten tips/all frames.',
        'Attraction: s_f=closest original triangle-surface point to HUMAN h_f; c_f=1[||h_f-s_f||<0.03m].',
        'E_contact=100 sum_f c_f ||p_f-s_f||^2. The selected surface point is fixed within the frame solve.',
        'It is not distance from the current ROBOT point to its nearest surface, nor a sparse object sample.',
        'Across frames the source-selected target may move; there is no persistent material contact or hard no-slip/force-closure requirement.',
        'For an active tip, tracking+attraction =125||p-(0.2h+0.8s)||^2+constant.',
        'If h=s this is only an increase in direct tracking weight. Otherwise it also moves that target.',
        'This does not make attraction equivalent to uniformly increasing the graph-Laplacian weight.', '',
        'Bone: 0.1 times squared differences of 15 unit phalange segment directions, expressed in each hand palm frame; no object targets.',
        'Palm: 0.5 times squared differences of wrist-to-middle-MCP axis and palm normal (also uses index MCP); no object targets.',
        'Finger prior: 0.005||q_finger-q_quick_init||^2; temporal prior: 0.15||q-q_previous||^2.',
        'Arm prior: lambda||q_arm-q_Omni||^2, lambda=0.002 if any source tip of that hand has contact, else 0.05.',
        'Collision: native moving-arm/hand geometry vs object convex pieces, ground and relevant self pairs;',
        'signed distance d+Jdq+s >= -0.001m, s>=0, slack cost 0.5*100000*sum(s^2). Not hard zero penetration.', '',
        'SHARED FIVE: equal-clip averages, 1240 frames. Organizer omitted here only because three variants failed.',
        'Variant | coarse % | mean hand depth mm | hand >1mm % | hand >2mm % | contact F1 % | 1mm-gated F1 % | 5mm-gated F1 % | 10mm proximity F1 %']
    for variant, data in aggregates['shared_five'].items():
        m = data['clip_equal_means']
        keys = ['combined_pen_duration_pct', 'hand_object_pen_mean_mm', 'hand_object_gt1mm_pct',
                'hand_object_gt2mm_pct', 'contact_f1_pct', 'contact_f1_penetration_gated_pct',
                'contact_f1_penetration_gated5mm_pct', 'contact_f1_both_labels_10mm_pct']
        report.append(variant+' | '+' | '.join(f'{m[k]:.4f}' for k in keys))
    report.extend(['', 'Source human-tip to source-projected-surface offsets, on contact-labelled tip-frames:',
                   'Sequence | object samples | mean mm | median mm | >1mm %'])
    for r in sources:
        report.append(f"{r['sequence']} | {r['object_sample_count']} | {r['source_to_projected_target_mean_mm']:.3f} | {r['source_to_projected_target_median_mm']:.3f} | {r['source_to_projected_target_gt1mm_pct']:.2f}")
    full = aggregates['shared_five']['full']['pooled_frame_max_depth_quantiles_mm']
    report.extend(['', f"Full V2 shared-five pooled hand-depth median/P95/max: {full['50']:.4f}/{full['95']:.4f}/{full['100']:.4f} mm.",
                   'High 30mm proximity F1 does not establish precise, penetration-free grasping. Many shallow depths sit just above the 1mm soft allowance.',
                   'The strict gate applies to an entire hand, amplifying even one proximal/fingertip geometry collision; it is not per-tip signed penetration.',
                   'The shared-five coarse full/no_contact/core figures flag the same six dining-chair frames; all other shared clips have zero coarse violations.',
                   'Full V2 organizer, excluded from shared-five averages: mean hand depth 4.1888mm; max 34.0079mm; >5mm duration 29.956%; 1mm-gated F1 10.8893%.',
                   'No_palm/no_arm_prior also have rare >20mm hand collisions in the shared cohort; see per-clip CSV/JSON.',
                   'These findings qualify any earlier recommendation based on 30mm F1 alone. No production settings changed.', '',
                   'Complete per-clip statistics, threshold sweeps, both cohorts, source paths and SHA256 provenance: objective_audit.json.'])
    (C.OUT/'objective_audit.txt').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(clips_audited=len(rows), shared_cohort=shared,
                          source_offsets=source_summary, report=str(C.OUT/'objective_audit.txt')), indent=2))


if __name__ == '__main__':
    main()
