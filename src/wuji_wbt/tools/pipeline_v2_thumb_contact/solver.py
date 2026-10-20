"""Compare contact priority with a geometry-aware replacement of the side loss.

The baseline side loss tests the tip reference point only. The support variant
tests the minimum signed plane clearance of the existing distal collision
convex. Its outward direction and finite patch are still source-only. No pad
point, new correspondence, altered collision mesh, or extra weight is used.
"""
from config import *
implementation = module('thumb_signed_solver', SIGNED_CODE / 'solver.py')
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

    def support_side(self, target, side, finger, derivatives=True):
        geom = int(self.tipgeom[side, finger])
        t = target['frame']
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
        r, J = super().additional(q, target, force_all)
        active = np.argwhere(self.data['contact'][target['frame']])
        # Parent order: (direction, surface gap) per active tip, then one signed
        # material residual per active tip. Both inherited flags remain enabled.
        assert len(r) == 3*len(active)
        for index, (side, finger) in enumerate(active):
            side, finger = int(side), int(finger)
            if self.experiment_variant == 'priority' and finger in (0, 1):
                ids = [2*index, 2*index+1, 2*len(active)+index]
                r[ids] *= 2.; J[ids] *= 2.  # four times squared contact cost
            elif self.experiment_variant == 'support':
                distance, gradient = self.support_side(target, side, finger)
                residual = distance + self.cfg.collision_tolerance
                weight = np.sqrt(EXTRA['direction_weight'])
                r[2*index] = weight*min(residual, 0.)
                J[2*index] = weight*gradient if residual < 0. else np.zeros(self.n)
        return r, J


implementation.original.Refiner = Refiner
parent_load = implementation.load
target_at = implementation.target_at


def load(seq, variant):
    result = parent_load(seq, 'signed')
    result[3].experiment_variant = 'baseline' if variant == 'both' else variant
    return result


implementation.original.load = load


def main(seq, variant):
    implementation.main(seq, variant)
    path = destination(seq, variant) / 'run.json'; run = read(path)
    run['signed_contact']['enabled'] = True
    run['thumb_contact_variant'] = variant
    run['thumb_contact_change'] = dict(priority='Four times existing direction, surface-gap, and signed-recovery costs on both thumb and index digits; all human tracking weights unchanged.',
        support='Replace reference-point halfspace residual by min(0, min_vertex n dot (v-anchor) + collision_tolerance), with the same direction coefficient. Source direction and regional patch unchanged.')[variant]
    write(path, run)


if __name__ == '__main__':
    main(*sys.argv[1:])
