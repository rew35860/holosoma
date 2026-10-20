"""Follow-up on the geometry-side correction; keep evaluation targets frozen."""
from config import *
parent = module('thumb_parent_solver', PARENT_CODE / 'solver.py')
np, mujoco = parent.np, parent.mujoco


class Refiner(parent.Refiner):
    def __init__(self, *args):
        super().__init__(*args)
        self.followup_variant = 'baseline'

    def additional(self, q, target, force_all=False):
        r, J = super().additional(q, target, force_all)
        active = np.argwhere(self.data['contact'][target['frame']])
        for index, (side, finger) in enumerate(active):
            if self.followup_variant == 'combined' and finger in (0, 1):
                ids = [2*index, 2*index+1, 2*len(active)+index]
                r[ids] *= 2.; J[ids] *= 2.
            elif self.followup_variant == 'touch' and side == 1 and finger == 0:
                # Existing r = sqrt(w)*(gap-source_gap). Change only this
                # optimization target to zero; do not mutate source data.
                r[2*index+1] += np.sqrt(EXTRA['gap_weight'])*self.data['source_skin_gap'][target['frame'], side, finger]
        return r, J


parent.implementation.original.Refiner = Refiner
parent_load = parent.load
target_at = parent.target_at


def load(seq, variant):
    result = parent_load(seq, 'baseline' if variant in ('baseline', 'both') else 'support')
    result[3].followup_variant = variant
    return result


parent.implementation.original.load = load


def main(seq, variant):
    parent.implementation.original.main(seq, variant)
    path = destination(seq, variant) / 'run.json'; run = read(path)
    run['contact_followup'] = dict(variant=variant,
        description=read(OUT / 'protocol.json')['variants'][variant],
        source_targets_untouched_for_evaluation=True, signed_recovery_enabled=True)
    write(path, run)


if __name__ == '__main__': main(*sys.argv[1:])
