"""Identical-settings diagnostic retry; records exact failing frame, never replaces a trial."""
import sys
from config import *
sys.path.insert(0,str(RET/'retarget_pipeline'))
import pipeline_v2 as solver


def main(variant,seq):
    dest=OUT/'retries'/variant/seq;dest.mkdir(parents=True,exist_ok=True)
    job=inputs(seq);original=solver.CoupledRefiner.frame_target;frame=[-1]
    def tracked(self,*args,**kwargs):
        frame[0]+=1
        return original(self,*args,**kwargs)
    solver.CoupledRefiner.frame_target=tracked
    sys.argv=[str(SOLVER),'--canonical',str(job['canonical']),'--body',str(job['body']),
        '--body-model',str(job['body_model']),'--model',str(job['model']),'--object',job['object'],
        '--settings',str(OUT/'settings'/f'{variant}.json'),'--output',str(dest)]
    result=dict(sequence=seq,variant=variant,settings=settings(variant),solver_sha256=sha(SOLVER),
        note='Identical objective/settings/inputs; wrapper only counts calls to frame_target. Original failed trial remains failed.')
    try:
        solver.main();result.update(state='success')
    except Exception as exc:
        result.update(state='failed',error=repr(exc),frame_zero_based=frame[0])
    write(dest/'retry.json',result)


if __name__=='__main__':main(sys.argv[1],sys.argv[2])
