"""Frozen V2 metrics and full-body collision audit for every completed solve."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
from config import *
import solver
def main(seq):
    module('cross_geometry_eval',GEOMETRY_CODE/'evaluate.py').main(seq,RUN_VARIANT)
    info,z,data,ref,body,init=solver.load(seq,'mujoco')
    audit=module('cross_fullbody_eval',CODE.parent/'pipeline_v2_shelf_body_collision/results.py')
    Q=np.load(destination(seq)/'final_v2.npz')['qpos'];stats,frames=audit.audit(ref,Q,info['object'])
    write(destination(seq)/'whole_body_collision.json',stats)
    np.savez_compressed(destination(seq)/'whole_body_collision.npz',depths_m=frames)
if __name__=='__main__':run_stage('evaluate',main,sys.argv[1])
