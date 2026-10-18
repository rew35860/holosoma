"""Resolve original and added matched clips without modifying the 120-clip archive."""
from pathlib import Path

WORK = Path(__file__).resolve().parents[4]
ROOT = WORK.parent
OUT = ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_pilot_20261018'
ORIGINAL_BATCH = ROOT/'holosoma/eval_out/ref_candidates/batch_20261011_current_120'
ADDITIONAL = ('push_table_forward_with_both_hands-039',
              'pick_up_and_carry_organizer_medium_while_walking-096')

def baseline_for(seq):
    return OUT/'additional_v1' if seq in ADDITIONAL else ORIGINAL_BATCH

def source_pkl(seq):
    for root in (ORIGINAL_BATCH/'preprocessed',ROOT/'Data/humoto_data'):
        p = root/'humoto_upbone_pkl'/seq/f'{seq}.pkl'
        if p.exists():return p
    raise FileNotFoundError(seq)
