"""Identical target definitions on each dataset's actual human surface."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from config import *
implementation=module('cross_prepare',CODE.parent/'pipeline_v2_humoto_smplx/prepare.py')
def main(seq):
    implementation.main(seq)
    path=OUT/'targets'/f'{seq}.json';meta=read(path)
    meta['source_representation']=row(seq)['dataset']+': '+read(OUT/'protocol.json')['datasets'][row(seq)['dataset']]
    meta['source_masks_recomputed_from_smplx']=row(seq)['dataset']=='hiphi'
    meta['caveat']='Source proximity labels, not tactile ground truth; same definitions, different source rigs across datasets.'
    write(path,meta)
    module('cross_geometry_audit',CODE/'audit_geometry.py').main(seq)
if __name__=='__main__':run_stage('prepare',main,sys.argv[1])
