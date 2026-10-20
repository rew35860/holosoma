"""Persistent bounded CPU batch; original body motion is reused, never rerun."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from config import *

def command(script,args,log):
    start=time.perf_counter()
    with Path(log).open('w') as stream:
        subprocess.run([PY,str(CODE/script),*args],cwd=WORK,stdout=stream,stderr=subprocess.STDOUT,check=True)
    return time.perf_counter()-start

def one(seq,variant):
    dest=destination(seq,variant);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'completed.json').exists():return read(dest/'completed.json')
    times={}
    if not (dest/'final_v2.npz').exists():times['solve_process_s']=command('solver.py',[seq,variant],dest/'solver.log')
    times['evaluation_s']=command('evaluate.py',[seq,variant],dest/'evaluation.log')
    result=dict(sequence=seq,variant=variant,state='complete',timings=times)
    write(dest/'completed.json',result);return result

def main():
    start=time.perf_counter();protocol=read(OUT/'protocol.json')
    for group in ('input_sha256','source_sha256','target_sha256'):
        for path,expected in protocol[group].items():assert sha(path)==expected,path
    write(OUT/'status.json',dict(state='numerical_checks',completed=0,total=8))
    command('preflight.py',[],OUT/'preflight.log')
    command('report.py',[],OUT/'report.log')
    # Interleave clips so both objects progress; only two CPU solve processes.
    jobs=[(seq,variant) for variant in REMOVALS for seq in COHORT];done=[];errors=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(one,*job):job for job in jobs}
        while pending:
            finished,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
            for future in finished:
                seq,variant=pending.pop(future)
                try:done.append(future.result())
                except Exception as exc:errors.append(dict(sequence=seq,variant=variant,error=repr(exc)))
            write(OUT/'status.json',dict(state='running',completed=len(done),total=8,errors=errors,
                elapsed_seconds=time.perf_counter()-start,pending=[dict(sequence=s,variant=v) for s,v in pending.values()]))
            if finished:command('report.py',[],OUT/'report.log')
    if errors:raise RuntimeError(errors)
    command('sensitivity.py',[],OUT/'sensitivity.log')
    write(OUT/'status.json',dict(state='rendering',completed=8,total=8,elapsed_seconds=time.perf_counter()-start))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(command,'render.py',[seq],OUT/f'render_{seq}.log') for seq in COHORT]
        render_times=[f.result() for f in futures]
    command('report.py',[],OUT/'report.log')
    command('validate.py',[],OUT/'validation.log')
    write(OUT/'status.json',dict(state='complete',completed=8,total=8,errors=[],
        elapsed_seconds=time.perf_counter()-start,render_seconds=render_times,validation_passed=True))
    command('report.py',[],OUT/'report.log')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status=read(OUT/'status.json');status.update(state='failed',error=repr(exc));write(OUT/'status.json',status);raise
