"""Bounded persistent organizer pilot: solve, evaluate, render, report."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import shutil
import subprocess
import time
from config import *


def main(finish_only=False):
    start=time.perf_counter();protocol=read(OUT/'protocol.json')
    for path,expected in protocol['input_sha256'].items():assert sha(path)==expected,path
    baseline=OUT/'variants/baseline/clips'/SEQ;baseline.mkdir(parents=True,exist_ok=True)
    for name in ('final_v2.npz','metrics.json','per_frame_metrics.npz'):
        shutil.copy2(PILOT/'clips'/SEQ/name,baseline/name)
    write(baseline/'completed.json',dict(reused=True))
    fixed=OUT/'variants/fixed_fingers/clips'/SEQ;fixed.mkdir(parents=True,exist_ok=True)
    for name in ('final_v2.npz','metrics.json','per_frame_metrics.npz','run.json'):
        shutil.copy2(FIXED/'variants/contact_ik/clips'/SEQ/name,fixed/name)
    write(fixed/'completed.json',dict(reused=True))
    stages={}
    sequence=[('audit','audit.py'),('report','report.py')] if finish_only else [('numerical_checks','preflight.py'),('solve','solver.py'),('evaluate','evaluate.py'),('audit','audit.py'),('render','render.py'),('report','report.py')]
    for stage,script in sequence:
        write(OUT/'status.json',dict(stage=stage,elapsed_seconds=time.perf_counter()-start,stages=stages))
        t=time.perf_counter()
        with (OUT/f'{stage}.log').open('w') as log:
            subprocess.run([PY,str(CODE/script)],cwd=WORK,check=True,stdout=log,stderr=subprocess.STDOUT)
        stages[stage]=time.perf_counter()-t
        if stage=='solve':write(OUT/'variants/contact_ik/clips'/SEQ/'completed.json',dict(completed_trajectory=True,feasibility_in='run.json'))
    for path,expected in protocol['input_sha256'].items():assert sha(path)==expected,path
    write(OUT/'status.json',dict(stage='complete',elapsed_seconds=time.perf_counter()-start,stages=stages,inputs_unchanged=True,finish_only=finish_only))
    if not finish_only:
        with (OUT/'validate.log').open('w') as log:
            subprocess.run([PY,str(CODE/'validate.py')],cwd=WORK,check=True,stdout=log,stderr=subprocess.STDOUT)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--finish-only',action='store_true')
    try:main(parser.parse_args().finish_only)
    except Exception as exc:
        status=read(OUT/'status.json');status.update(stage='failed',error=repr(exc))
        write(OUT/'status.json',status);raise
