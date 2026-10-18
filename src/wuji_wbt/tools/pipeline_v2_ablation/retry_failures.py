"""Retry failed primary runs once, unchanged, retaining their original failure status."""
import subprocess
import time
from config import *

def main():
    original=read(OUT/'status.json')
    assert original['state']!='running'
    write(OUT/'primary_batch_status.json',original)
    records=[]
    for failed in original['errors']:
        v,seq=failed['variant'],failed['sequence'];job=inputs(seq)
        dest=OUT/'retries'/v/'clips'/seq;dest.mkdir(parents=True,exist_ok=True)
        assert read(OUT/'settings'/f'{v}.json')==settings(v)
        assert sha(SOLVER)==read(OUT/'input_manifest.json')['solver_sha256']
        args=[PY,str(CODE/'replay.py'),'--canonical',str(job['canonical']),
              '--body',str(job['body']),'--body-model',str(job['body_model']),
              '--model',str(job['model']),'--object',job['object'],
              '--settings',str(OUT/'settings'/f'{v}.json'),'--output',str(dest)]
        start=time.perf_counter()
        with (dest/'refinement.log').open('w') as log:
            result=subprocess.run(args,cwd=RET,stdout=log,stderr=subprocess.STDOUT)
        r=dict(variant=v,sequence=seq,returncode=result.returncode,seconds=time.perf_counter()-start,
               settings_identical=True,solver_identical=True,original_failure_preserved=True)
        if (dest/'failure_diagnostic.json').exists():r['diagnostic']=read(dest/'failure_diagnostic.json')
        records.append(r)
        write(OUT/'retry_summary.json',dict(retries=records,primary_failures_unchanged=True))
    original.update(state='complete_with_model_failures' if original['errors'] else 'complete',
                    attempted=original['completed']+len(original['errors']),
                    failed=len(original['errors']),retries_completed=len(records),
                    note='All planned primary runs attempted. Failures retained; unchanged retries are diagnostics, not replacement successes.')
    write(OUT/'status.json',original)

if __name__=='__main__':main()
