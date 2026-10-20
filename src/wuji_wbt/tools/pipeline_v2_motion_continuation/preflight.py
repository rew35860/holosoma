"""Verify the control, invariant inputs and bounded predictor/continuation."""
from config import *
from solver import *

def main():
    results=[]
    for seq in COHORT:
        info,z,data,ref,body,init=load(seq,'baseline')
        Q=np.load(PREVIOUS/'variants/geometry/clips'/seq/'final_v2.npz')['qpos']
        t=111 if 'organizer' in seq else 82
        previous=Q[t-1];q=init[t].copy();q[ref.qadr]=previous[ref.qadr]
        target=target_at(ref,z,data,t)
        control,record=ref.solve_stage(q,target,body[t],init[t],previous)
        assert np.array_equal(control,Q[t]),'Original frame solve must match exactly'
        assert np.array_equal(mix(body[t-1],body[t],0.),body[t-1])
        assert np.array_equal(mix(body[t-1],body[t],1.),body[t])
        predicted,detail=ref.compensate(q,previous,data['contact'][t],data['contact'][t-1],1.)
        assert np.abs(predicted[ref.qadr]-previous[ref.qadr]).max()<=ref.cfg.joint_rate_limit+1e-8
        fixed=np.setdiff1d(np.arange(ref.m.nq),ref.qadr)
        assert np.array_equal(predicted[fixed],q[fixed])
        d0=ref.depth_map(q);d1=ref.depth_map(predicted)
        continuation={}
        for compensated in (False,True):
            test,rr=ref.continue_frame(z,data,t,body,init,previous,compensated)
            assert np.array_equal(test[fixed],body[t,fixed])
            assert np.abs(test[ref.qadr]-previous[ref.qadr]).max()<=ref.cfg.joint_rate_limit+1e-6
            assert abs(sum(a['end']-a['start'] for a in rr['attempts'] if a['accepted'])-1)<1e-10
            continuation[str(compensated)]=dict(substeps=rr['substeps'],guard_unresolved=rr['guard_unresolved'],resource_cap_fallback=rr['resource_cap_fallback'],
                max_registered_depth_mm=max(ref.depth_map(test).values(),default=0)*1000)
        results.append(dict(sequence=seq,frame=t,control_exact=True,predictor=detail,
            original_warm_max_registered_depth_mm=max(d0.values())*1000,
            predicted_max_registered_depth_mm=max(d1.values())*1000,continuation=continuation))
        print(results[-1],flush=True)
    write(OUT/'preflight.json',dict(passed=True,checks=results))

if __name__=='__main__':main()
