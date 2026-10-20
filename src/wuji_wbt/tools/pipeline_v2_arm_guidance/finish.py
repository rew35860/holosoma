"""Finish the visual/comfort audit after the bounded optimization batch exits."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess
import time
from config import *


def command(args,log):
    with log.open('a') as f:
        subprocess.run([PY,*map(str,args)],cwd=CODE,stdout=f,stderr=subprocess.STDOUT,check=True)


def one(seq):
    log=OUT/'logs'/f'{seq}_visual_audit.log'
    for v in REMOVALS:
        dest=OUT/'variants'/v/'clips'/seq
        if (dest/'completed.json').exists() and not (dest/'comfort.json').exists():
            command([CODE/'diagnostics.py',v,seq],log)
        if (dest/'completed.json').exists():
            command([CODE/'collision_audit.py',v,seq],log)
    command([CODE/'graphs.py',seq],log)
    command([CODE/'render.py',seq],log)
    return seq


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    while not (OUT/'status.json').exists() or read(OUT/'status.json')['state']=='running':time.sleep(10)
    (OUT/'logs').mkdir(exist_ok=True)
    start=time.perf_counter();done=[];errors=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(one,seq):seq for seq in COHORT}
        for future in as_completed(pending):
            try:done.append(future.result())
            except Exception as exc:errors.append(dict(sequence=pending[future],error=repr(exc)))
            write(OUT/'visual_status.json',dict(state='running',complete=done,errors=errors,elapsed_seconds=time.perf_counter()-start))
            command([CODE/'report.py'],OUT/'logs/report.log')
    write(OUT/'visual_status.json',dict(state='failed' if errors else 'complete',complete=done,errors=errors,elapsed_seconds=time.perf_counter()-start))
    command([CODE/'report.py'],OUT/'logs/report.log')
    if errors:raise RuntimeError(errors)


if __name__=='__main__':main()
