"""Run the frozen evaluator on added baseline clips using isolated paths."""
import importlib.util
from pathlib import Path
import sys
from catalog import WORK, baseline_for
seq=sys.argv[1]
spec=importlib.util.spec_from_file_location('frozen_eval',WORK/'src/wuji_wbt/tools/ref_candidates_metrics/evaluate.py')
E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
E.BATCH=baseline_for(seq);E.OUT=E.BATCH/'evaluation_v1'
E.evaluate(seq)
