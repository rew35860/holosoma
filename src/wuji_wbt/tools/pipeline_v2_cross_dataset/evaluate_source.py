"""Original metric formulas; use the declared source of this dataset."""
from config import *
ROOT_OUT=OUT
implementation=module('cross_frozen_metrics',CODE.parent/'pipeline_v2_humoto_smplx/evaluate_source.py')
def main(seq):
    implementation.main(seq)
    path=destination(seq)/'metrics.json';meta=read(path)
    meta['source']=read(ROOT_OUT/'protocol.json')['datasets'][row(seq)['dataset']]
    meta['definitions']='Existing V2 formulas/thresholds, fixed labels within each clip across all candidates; cross-dataset labels derive from different source rigs.'
    write(path,meta)
