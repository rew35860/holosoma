"""Supplemental object penetration by registered robot collision link."""
import csv
import numpy as np
import mujoco
from config import *
from solver import load


def main(seq):
    _, _, _, ref, _, _ = load(seq, 'surface')
    robot_geoms = np.unique(ref.pairs[ref.pairs[:, 2] == 0, 0])
    body_ids = np.unique(ref.m.geom_bodyid[robot_geoms])
    body_index = {int(b): i for i, b in enumerate(body_ids)}
    names = [mujoco.mj_id2name(ref.m, mujoco.mjtObj.mjOBJ_BODY, int(b)) for b in body_ids]
    rows = []
    folder = OUT / 'link_audit'; folder.mkdir(exist_ok=True)
    for variant in REMOVALS:
        Q = np.load(destination(seq, variant) / 'final_v2.npz')['qpos']
        depths = np.zeros((len(Q), len(body_ids)))
        for t, q in enumerate(Q):
            distances, _, ids = ref.collisions(q, False)
            for distance, pair_index in zip(distances, ids):
                ga, _, category = ref.pairs[pair_index]
                if category != 0 or distance >= 0:
                    continue
                j = body_index[int(ref.m.geom_bodyid[ga])]
                depths[t, j] = max(depths[t, j], -distance)
        for j, name in enumerate(names):
            rows.append(dict(sequence=seq, variant=variant, robot_link=name,
                maximum_depth_mm=float(depths[:, j].max()*1000),
                mean_frame_depth_mm=float(depths[:, j].mean()*1000),
                above_1mm_frame_pct=float(np.mean(depths[:, j] > .001)*100),
                worst_frame=int(np.argmax(depths[:, j]))))
        np.savez_compressed(folder / f'{seq}_{variant}.npz', links=np.asarray(names), depth_m=depths)
    with (folder / f'{seq}.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write(folder / f'{seq}.json', dict(sequence=seq,
        definition='Maximum signed penetration per frame over each link’s registered object collision pairs; all frames; same Coal queries in both variants.',
        worst_by_variant={v: sorted([r for r in rows if r['variant'] == v], key=lambda r: -r['maximum_depth_mm'])[:8] for v in REMOVALS}))
