"""Audit intended side, physical fingertip extent, and cavity accessibility."""
import numpy as np
import coal
from config import *
from solver import load, target_at


def audit(seq, variant):
    info, z, data, ref, body, init = load(seq, variant)
    Q = np.load(destination(seq, variant) / 'final_v2.npz')['qpos']
    shape = data['contact'].shape
    center_side = np.full(shape, np.nan); support = np.full(shape, np.nan)
    gap = np.full(shape, np.nan); material = np.full(shape, np.nan)
    robot_local = np.zeros((len(Q), 2, 21, 3))
    for t, q in enumerate(Q):
        target = target_at(ref, z, data, t); ref.fk(q)
        p = ref.keypoints(False)
        robot_local[t] = (p-target['object_pose'][:3]) @ target['R']
        for s, f in np.argwhere(data['contact'][t]):
            normal = data['direction_local'][t, s, f]; anchor = data['anchor_local'][t, s, f]
            center_side[t, s, f] = (robot_local[t, s, 4+4*f]-anchor) @ normal
            support[t, s, f], _ = ref.support_side(target, int(s), int(f), False)
            gap[t, s, f] = ref.gap_query(target, int(s), int(f), False)[0]
            material[t, s, f] = ref.material_query(int(s), int(f), False)[0]
    mask = data['contact'][:, 1, 0]
    others = data['contact'][:, 1, 1:]
    # A stricter intended-patch indicator supplements, never replaces, the old F1.
    touch = (center_side >= -.001) & (gap <= .002) & (material >= -.001)
    result = dict(sequence=seq, variant=variant, frames=len(Q), right_thumb_contact_frames=int(mask.sum()),
        right_thumb_correct_side_pct=float(100*np.mean(center_side[:, 1, 0][mask] >= -.001)),
        right_thumb_wrong_side_mean_mm=float(np.maximum(-center_side[:, 1, 0][mask], 0).mean()*1000),
        right_thumb_geometry_side_pct=float(100*np.mean(support[:, 1, 0][mask] >= -.001)),
        right_thumb_patch_gap_mm=float(gap[:, 1, 0][mask].mean()*1000),
        right_thumb_max_material_pen_mm=float(np.maximum(-material[:, 1, 0][mask], 0).max()*1000),
        right_thumb_intended_contact_pct=float(touch[:, 1, 0][mask].mean()*100),
        right_other_fingers_correct_side_pct=float(100*np.mean(center_side[:, 1, 1:][others] >= -.001)),
        right_other_fingers_patch_gap_mm=float(gap[:, 1, 1:][others].mean()*1000),
        definitions=dict(side='Existing source exterior direction; correct reference side >= -1 mm. For this organizer the thumb target normal points inward and the four finger target normals outward.',
            geometry_side='Minimum plane coordinate of every vertex of the existing distal collision convex >= -1 mm.',
            intended_contact='Reference on intended side within 1 mm, finite source patch gap <=2 mm, signed distal-tip material penetration <=1 mm.',
            scope='All existing source-contact frames; original labels and general evaluator unchanged.'))
    folder = OUT / 'audit'; folder.mkdir(exist_ok=True)
    write(folder / f'{variant}.json', result)
    np.savez_compressed(folder / f'{variant}.npz', center_side=center_side, support_side=support,
        patch_gap=gap, material=material, robot_local=robot_local, source_contact=data['contact'])
    if variant == 'baseline':
        t = 120; target = target_at(ref, z, data, t); ref.fk(Q[t])
        anchor = target['R'] @ data['anchor_local'][t, 1, 0] + target['object_pose'][:3]
        normal = target['R'] @ data['direction_local'][t, 1, 0]
        sphere = coal.Sphere(.001); probes = []
        for offset in (.002, .005, .01, .02, .03):
            position = anchor + normal*offset
            distances = []
            for obj in ref.material[int(ref.tipgeom[1, 0])]:
                answer = coal.DistanceResult()
                distance = float(coal.distance(sphere, coal.Transform3s(np.eye(3), position),
                    ref.queries.shapes[int(obj)], coal.Transform3s(ref.d.geom_xmat[obj].reshape(3, 3), ref.d.geom_xpos[obj]),
                    ref.queries.request, answer))
                distances.append(distance)
            probes.append(dict(offset_into_cavity_mm=offset*1000, sphere_radius_mm=1.,
                               min_signed_distance_mm=min(distances)*1000))
        # Differentiate the support residual away from witness switches.
        errors = []
        for s, f in ((1, 0), (1, 1), (0, 0)):
            ref.fk(Q[t]); _, analytic = ref.support_side(target, s, f)
            for col in np.r_[ref.arm_columns[s], ref.finger_columns[s][:4]]:
                plus = Q[t].copy(); minus = Q[t].copy()
                plus[ref.qadr[col]] += 1e-6; minus[ref.qadr[col]] -= 1e-6
                ref.fk(plus); dp = ref.support_side(target, s, f, False)[0]
                ref.fk(minus); dm = ref.support_side(target, s, f, False)[0]
                errors.append(abs((dp-dm)/2e-6-analytic[col]))
        assert max(errors) < 1e-5, max(errors)
        records = read(destination(seq, variant) / 'iterations.json')
        write(OUT / 'diagnosis.json', dict(target_frame=t,
            right_source_normals=data['face_normal_local'][t, 1].tolist(),
            right_source_anchors=data['anchor_local'][t, 1].tolist(),
            right_thumb_center_plane_mm=float(center_side[t, 1, 0]*1000),
            right_thumb_geometry_plane_mm=float(support[t, 1, 0]*1000),
            right_thumb_patch_gap_mm=float(gap[t, 1, 0]*1000), cavity_probes=probes,
            support_gradient_max_abs_error=max(errors),
            line_search_stopped_frames=sum(r['linesearch_stopped'] for r in records),
            interpretation='The source assigns opposite wall faces to thumb and other digits. The current thumb reference is on the wrong side during carrying. A center-side soft loss does not enforce the side of the entire contact geometry, and a local SQP path can remain trapped across a thin wall. Sphere probes test cavity accessibility, not full-thumb reachability.'))
    return result


if __name__ == '__main__':
    print(audit(*sys.argv[1:]))
