"""Change only signed collision distance/witness queries; retain surface-gap queries.

load() intentionally always returns the common Coal evaluator. Only main() swaps
the optimization backend, preventing self-evaluation with different yardsticks.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1',
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import time
import shutil
import numpy as np
from config import *

implementation = module('backend_current_solver', IMPLEMENTATION_CODE/'solver.py')
load, target_at, mujoco = implementation.load, implementation.target_at, implementation.mujoco

class MujocoQueries:
    """Native mj_geomDistance with the same exact-pose cache as the Coal path."""
    def __init__(self, original):
        self.m = original.m
        self.shapes = original.shapes  # Unchanged unsigned regional surface-gap objective.
        self.configuration, self.cache = None, {}
        self.events, self.log_path = [], None

    def query(self, data, a, b, cutoff=.025):
        key = data.qpos.tobytes()
        if key != self.configuration:
            self.configuration, self.cache = key, {}
        pair = (int(a), int(b), float(cutoff))
        if pair not in self.cache:
            witness = np.zeros(6)
            distance = float(mujoco.mj_geomDistance(self.m, data, pair[0], pair[1], cutoff, witness))
            delta = witness[:3] - witness[3:]
            normal = np.sign(distance)*delta/max(float(np.linalg.norm(delta)), 1e-12)
            if not np.isfinite(distance) or not np.isfinite(witness).all():
                raise RuntimeError(f'Nonfinite MuJoCo distance for {pair}')
            self.cache[pair] = distance, witness[:3].copy(), witness[3:].copy(), normal
        return self.cache[pair]

def main(seq, variant=SELECTED_VARIANT):
    assert variant in VARIANTS
    start = time.perf_counter()
    info, z, data, ref, body, init = load(seq, variant)
    if variant == 'mujoco':
        ref.queries = MujocoQueries(ref.queries)
    dest = destination(seq, variant)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(info['initialization'], dest/'finger_initialization.npz')
    ref.queries.log_path = dest/'collision_query_retries.json'
    records, Q, previous = [], body.copy(), None
    begin, cpu_begin = time.perf_counter(), time.process_time()
    for t in range(len(Q)):
        q = init[t].copy()
        if previous is not None:
            q[ref.qadr] = previous[ref.qadr]
        predictor = None
        if t:
            q, predictor = ref.compensate(q, previous, data['contact'][t], data['contact'][t-1], 1.)
        target = target_at(ref, z, data, t)
        q, record = ref.solve_stage(q, target, body[t], init[t], previous, first=t == 0)
        record.update(frame=t, predictor=predictor)
        Q[t], previous = q, q.copy()
        records.append(record)
        if t % 20 == 0 or t == len(Q)-1:
            write(dest/'status.json', dict(state='solving', frames=t+1, total=len(Q),
                                           elapsed_seconds=time.perf_counter()-start))
            print(seq, variant, t+1, '/', len(Q), flush=True)
    elapsed, cpu_elapsed = time.perf_counter()-begin, time.process_time()-cpu_begin
    fixed = np.setdiff1d(np.arange(ref.m.nq), ref.qadr)
    assert np.array_equal(Q[:, fixed], body[:, fixed]) and np.isfinite(Q).all()
    assert np.all(Q[:, ref.qadr] >= ref.lo-1e-7) and np.all(Q[:, ref.qadr] <= ref.hi+1e-7)
    assert np.abs(np.diff(Q[:, ref.qadr], axis=0)).max() <= .2+1e-6
    control_matches = None
    if variant == 'coal':
        control_matches = np.array_equal(Q, np.load(SHARED/'variants/motion_init/clips'/seq/'final_v2.npz')['qpos'])
        assert control_matches, 'Coal must reproduce the current archived version exactly'
    np.savez_compressed(dest/'final_v2.npz', qpos=Q, fps=30, human_joints=z['human_joints_scaled'])
    write(dest/'iterations.json', records)
    write(dest/'run.json', dict(sequence=seq, variant=variant, frames=len(Q),
        settings=vars(ref.cfg), extra=EXTRA, options=OPTIONS,
        signed_recovery=dict(enabled=False, weight=0), geometry_side=True,
        motion_compensated_initialization=True, thumb_index_contact_multiplier=1,
        hard_side_constraint=False, substeps=False, collision_backend=variant,
        surface_gap_backend='Unchanged Coal unsigned convex-to-triangle-patch queries',
        mujoco_version=mujoco.__version__, mujoco_options=dict(
            disableflags=int(ref.m.opt.disableflags), enableflags=int(ref.m.opt.enableflags),
            ccd_iterations=int(ref.m.opt.ccd_iterations), ccd_tolerance=float(ref.m.opt.ccd_tolerance)),
        joint_refinement_seconds=elapsed, joint_refinement_cpu_seconds=cpu_elapsed,
        total_seconds=time.perf_counter()-start,
        solver_seconds=sum(s['solver_seconds'] for r in records for s in r['steps']),
        qp_count=sum(len(r['steps']) for r in records),
        linesearch_stopped_frames=sum(r['linesearch_stopped'] for r in records),
        coal_control_matches_archived_exactly=control_matches, fixed_body_object_exact=True,
        target_sha256=sha(OUT/'targets'/f'{seq}.npz'),
        initialization_sha256=sha(info['initialization']), model_sha256=sha(info['model']),
        collision_pairs_sha256=__import__('hashlib').sha256(ref.pairs.tobytes()).hexdigest(),
        final_sha256=sha(dest/'final_v2.npz')))
    write(dest/'status.json', dict(state='solved', frames=len(Q)))

if __name__ == '__main__':
    main(*sys.argv[1:])
