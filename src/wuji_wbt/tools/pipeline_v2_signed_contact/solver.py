"""Retain regional attraction; recover penetration using signed solid geometry.

An open patch has no material interior. Do not attach an arbitrary sign to its
unsigned distance or replace it with nearest-object attraction, which can move
contact to a different region. Add 256 sum(min(d_material, 0)**2) instead. The
piecewise-convex minimum is not an exact SDF of the nonconvex union.
"""
from config import *
original = module('signed_contact_original_solver', BASE_CODE / 'solver.py')
np, mujoco = original.np, original.mujoco
BaseRefiner = original.Refiner
CoupledRefiner = original.CoupledRefiner
target_at = original.target_at


class CachedQueries(original.CheckedQueries):
    """Reuse exact signed queries at identical configurations only."""
    def __init__(self, model):
        super().__init__(model)
        self.configuration = None
        self.cache = {}

    def query(self, data, a, b, cutoff=.025):
        key = data.qpos.tobytes()
        if key != self.configuration:
            self.configuration, self.cache = key, {}
        pair = (int(a), int(b), float(cutoff))
        if pair not in self.cache:
            self.cache[pair] = super().query(data, a, b, cutoff)
        return self.cache[pair]


class Refiner(BaseRefiner):
    def __init__(self, *args):
        super().__init__(*args)
        self.queries = CachedQueries(self.m)
        self.numerical_fallbacks = []
        self.material = {}
        for tip in self.tipgeom.ravel():
            self.material[int(tip)] = np.unique(self.pairs[(self.pairs[:, 2] == 0) &
                                                        (self.pairs[:, 0] == tip), 1])
            assert len(self.material[int(tip)])

    def material_query(self, side, finger, derivatives=True):
        tip = int(self.tipgeom[side, finger])
        objects = self.material[tip]
        lower = np.linalg.norm(self.d.geom_xpos[objects] - self.d.geom_xpos[tip], axis=1)
        lower -= self.m.geom_rbound[objects] + self.m.geom_rbound[tip]
        best, selected, witness, normal = np.inf, None, None, None
        for j in np.argsort(lower):
            if lower[j] >= best + 1e-12:
                continue
            obj = int(objects[j])
            distance, pa, _, n = self.queries.query(self.d, tip, obj)
            if distance < best:
                best, selected, witness, normal = distance, obj, pa, n
        assert selected is not None and np.isfinite(best)
        gradient = np.zeros(self.n)
        if derivatives:
            jac = np.zeros((3, self.m.nv))
            mujoco.mj_jac(self.m, self.d, jac, None, witness, int(self.m.geom_bodyid[tip]))
            gradient = normal @ jac[:, self.dadr]
        return float(best), gradient, (tip, selected)

    def recovery(self, q, target):
        self.fk(q)
        residuals, jacobians = [], []
        for side, finger in np.argwhere(self.data['contact'][target['frame']]):
            distance, gradient, _ = self.material_query(int(side), int(finger))
            residuals.append(np.sqrt(SIGNED_WEIGHT) * min(distance, 0.))
            jacobians.append(np.sqrt(SIGNED_WEIGHT) * gradient if distance < 0. else np.zeros(self.n))
        return np.asarray(residuals), np.asarray(jacobians).reshape(-1, self.n)

    def additional(self, q, target, force_all=False):
        r, J = super().additional(q, target, force_all)
        if self.variant != 'signed':
            return r, J
        extra, G = self.recovery(q, target)
        return np.r_[r, extra], np.vstack([J, G])

    def subproblem(self, q, residual, J, dist, gradient, previous):
        # Use the shared V2 solve: explicit tolerances, no objective rescaling.
        step, info = super().subproblem(q, residual, J, dist, gradient, previous)
        if info['accepted_feasible_fallback']:
            self.numerical_fallbacks.append(dict(frame=self.frame, **info))
            if self.queries.log_path:
                write(self.queries.log_path.parent / 'numerical_fallbacks.json', self.numerical_fallbacks)
        return step, info


original.Refiner = Refiner
original_load = original.load


def load(seq, variant):
    # The frozen evaluator requests 'both' to measure its original unsigned gap.
    # Map this diagnostic-only alias to the surface baseline, not signed loss.
    result = original_load(seq, 'surface' if variant == 'both' else variant)
    result[3].cfg.contact_weight = 0.
    return result


original.load = load

def main(seq, variant):
    original.main(seq, variant)
    path = destination(seq, variant) / 'run.json'
    run = read(path)
    run['signed_contact'] = dict(enabled=variant == 'signed', weight=SIGNED_WEIGHT,
        formula='sum_i 256 * min(d_material_i, 0)^2 over unchanged source-contact fingertips',
        geometry='Minimum signed separation over the existing fingertip/object convex pairs; not an exact nonconvex-union SDF',
        unsigned_patch_gap_retained=True, direction_unchanged=True)
    fallbacks = destination(seq, variant) / 'numerical_fallbacks.json'
    run['numerical_recovery'] = dict(count=len(read(fallbacks)) if fallbacks.exists() else 0,
        policy='Unscaled QP; gap abs/rel 1e-6, feasibility 1e-7; reduced defaults unchanged. Check original-unit cone feasibility for every returned step. For InsufficientProgress/MaxIterations also require an independent dual-bound gap certificate. No post-solve clipping.')
    write(path, run)


if __name__ == '__main__':
    main(*sys.argv[1:])
