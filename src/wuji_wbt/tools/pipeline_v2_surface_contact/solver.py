"""Reuse the checked solver; change only the legacy attraction coefficient."""
from config import *

original = module('surface_contact_original_solver', BASE_CODE / 'solver.py')
original_load = original.load
CoupledRefiner = original.CoupledRefiner
target_at = original.target_at


def load(seq, variant):
    result = original_load(seq, variant)
    if variant == 'surface':
        result[3].cfg.contact_weight = 0.0
    return result


original.load = load

if __name__ == '__main__':
    original.main(*sys.argv[1:])
