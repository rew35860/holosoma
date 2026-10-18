"""Bounded, resumable CPU rendering of all requested three-panel comparisons."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import subprocess
import time
from catalog import OUT

CODE=Path(__file__).resolve().parent
PY='/home/natcha/miniconda3/envs/omniretarget/bin/python'
def render(seq):
    result=OUT/'renders_human'/f'{seq}.json'
    if result.exists():return json.loads(result.read_text())
    with (OUT/f'render_human_{seq}.log').open('a') as f:
        subprocess.run([PY,str(CODE/'render_three_panel.py'),seq],stdout=f,stderr=subprocess.STDOUT,check=True)
    return json.loads(result.read_text())
def status(x):
    p=OUT/'render_human_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
if __name__=='__main__':
    start=time.perf_counter();seqs=json.loads((OUT/'cohort.json').read_text())['sequences'];done=[];errors=[]
    status(dict(state='running',completed=0,total=len(seqs)))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(render,s):s for s in seqs}
        for f in as_completed(futures):
            try:done.append(f.result());print('RENDERED',futures[f],flush=True)
            except Exception as exc:errors.append(dict(sequence=futures[f],error=repr(exc)))
            status(dict(state='running',completed=len(done),total=len(seqs),errors=errors))
    status(dict(state='complete' if not errors else 'failed',completed=len(done),total=len(seqs),
                wall_seconds=time.perf_counter()-start,errors=errors))
    if errors:raise RuntimeError(errors)
