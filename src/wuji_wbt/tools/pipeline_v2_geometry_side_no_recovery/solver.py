"""Restore geometry-side guidance without the signed-recovery objective.

Signed Coal collision checks, slack, and the shared checked QP remain unchanged.
Priority multiplies only side and gap squared costs for thumb/index on both hands.
No special thumb constraint is added. The existing support geometry/formula is
the same as pipeline_v2_thumb_contact; its signed residual is not inherited here.
"""
from config import *
implementation = module('geometry_no_recovery_signed_backend', SIGNED_CODE / 'solver.py')
np, mujoco = implementation.np, implementation.mujoco


class Refiner(implementation.Refiner):
    def __init__(self, *args):
        super().__init__(*args)
        self.experiment_variant = 'baseline'
        self.tip_vertices = {}
        for geom in self.tipgeom.ravel():
            mid = int(self.m.geom_dataid[geom])
            start, count = int(self.m.mesh_vertadr[mid]), int(self.m.mesh_vertnum[mid])
            self.tip_vertices[int(geom)] = self.m.mesh_vert[start:start+count].astype(float).copy()

    def recovery(self, *args, **kwargs):
        raise AssertionError('Signed-recovery objective is disabled for all experiment variants')

    def support_side(self, target, side, finger, derivatives=True):
        geom = int(self.tipgeom[side, finger]); t = target['frame']
        normal = target['R'] @ self.data['direction_local'][t, side, finger]
        anchor = target['R'] @ self.data['anchor_local'][t, side, finger] + target['object_pose'][:3]
        rotation = self.d.geom_xmat[geom].reshape(3, 3)
        vertices = self.tip_vertices[geom] @ rotation.T + self.d.geom_xpos[geom]
        distances = (vertices-anchor) @ normal
        index = int(np.argmin(distances))
        gradient = np.zeros(self.n)
        if derivatives:
            jac = np.zeros((3, self.m.nv))
            mujoco.mj_jac(self.m, self.d, jac, None, vertices[index], int(self.m.geom_bodyid[geom]))
            gradient = normal @ jac[:, self.dadr]
        return float(distances[index]), gradient

    def additional(self, q, target, force_all=False):
        # Explicitly call the pre-recovery residual builder: two rows/contact,
        # never the third signed-material-recovery row.
        r, J = implementation.BaseRefiner.additional(self, q, target, force_all)
        active = np.argwhere(self.data['contact'][target['frame']])
        assert len(r) == 2*len(active)
        for index, (side, finger) in enumerate(active):
            side, finger = int(side), int(finger)
            if self.experiment_variant in ('geometry', 'priority'):
                clearance, gradient = self.support_side(target, side, finger)
                violation = clearance + self.cfg.collision_tolerance
                weight = np.sqrt(EXTRA['direction_weight'])
                r[2*index] = weight*min(violation, 0.)
                J[2*index] = weight*gradient if violation < 0. else np.zeros(self.n)
            if self.experiment_variant == 'priority' and finger in (0, 1):
                r[2*index:2*index+2] *= 2.
                J[2*index:2*index+2] *= 2.
        return r, J


implementation.original.Refiner = Refiner
parent_load = implementation.load
target_at = implementation.target_at


def load(seq, variant):
    assert variant in REMOVALS
    result = parent_load(seq, 'surface')
    result[3].experiment_variant = variant if variant in VARIANTS else 'baseline'
    assert result[3].cfg.contact_weight == 0.
    return result


implementation.original.load = load


def main(seq, variant):
    assert variant in VARIANTS
    implementation.original.main(seq, variant)
    path = destination(seq, variant) / 'run.json'; run = read(path)
    run.update(signed_recovery=dict(enabled=False,weight=0.),
        geometry_side=variant != 'baseline', hard_side_constraint=False,
        thumb_index_contact_multiplier=4 if variant == 'priority' else 1,
        multiplier_scope='Side-direction and regional surface-gap costs only; thumb/index on both hands during source contact. Human fingertip tracking unchanged.')
    write(path, run)


if __name__ == '__main__':
    main(*sys.argv[1:])
