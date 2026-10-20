"""Verify isolated residual changes, frozen collision handling and support derivatives."""
import numpy as np
from config import *
from solver import load, target_at, implementation


def main():
    records=[]
    for seq in COHORT:
        _,z,data,ref,body,init=load(seq,'baseline')
        Q=np.load(BASELINE/'variants/signed/clips'/seq/'final_v2.npz')['qpos']
        active=np.flatnonzero(data['contact'].any(axis=(1,2)))
        for t in active[[0,len(active)//2,-1]]:
            t=int(t);q=Q[t];previous=Q[t-1] if t else None
            target=target_at(ref,z,data,t)
            ref.experiment_variant='baseline'
            rb,Jb=ref.additional(q,target)
            expected,Ge=implementation.BaseRefiner.additional(ref,q,target)
            assert np.array_equal(rb,expected) and np.array_equal(Jb,Ge)
            base_full,base_J=ref.residuals(q,target,body[t],init[t],previous)
            noncontact=len(base_full)-len(rb)
            collision=ref.collisions(q)
            contacts=np.argwhere(data['contact'][t])
            ref.experiment_variant='geometry';rg,Jg=ref.additional(q,target)
            assert np.array_equal(rg[1::2],rb[1::2]) and np.array_equal(Jg[1::2],Jb[1::2])
            errors=[]
            for index,(side,finger) in enumerate(contacts):
                side,finger=int(side),int(finger)
                ref.fk(q);d,G=ref.support_side(target,side,finger)
                assert abs(rg[2*index]-50*min(d+.001,0.))<1e-12
                for col in (ref.arm_columns[side][0],ref.finger_columns[side][4*finger]):
                    plus=q.copy();minus=q.copy();plus[ref.qadr[col]]+=1e-6;minus[ref.qadr[col]]-=1e-6
                    ref.fk(plus);dp=ref.support_side(target,side,finger,False)[0]
                    ref.fk(minus);dm=ref.support_side(target,side,finger,False)[0]
                    errors.append(abs((dp-dm)/2e-6-G[col]))
            assert max(errors,default=0.)<1e-5,(seq,t,max(errors))
            ref.experiment_variant='priority';rp,Jp=ref.additional(q,target)
            multipliers=np.repeat([2. if f in (0,1) else 1. for _,f in contacts],2)
            assert np.array_equal(rp,rg*multipliers) and np.array_equal(Jp,Jg*multipliers[:,None])
            for v in VARIANTS:
                ref.experiment_variant=v
                full,J=ref.residuals(q,target,body[t],init[t],previous)
                assert len(full)==len(base_full)
                assert np.array_equal(full[:noncontact],base_full[:noncontact])
                assert np.array_equal(J[:noncontact],base_J[:noncontact])
                assert all(np.array_equal(a,b) for a,b in zip(collision,ref.collisions(q)))
            assert type(ref).subproblem is implementation.Refiner.subproblem
            records.append(dict(sequence=seq,frame=t,active_contacts=len(contacts),
                residual_rows_per_contact=2,signed_recovery_rows=0,common_objectives_exact=True,
                collision_distances_gradients_pairs_exact=True,priority_cost_multiplier=4,
                support_gradient_max_error=max(errors,default=0.),no_extra_qp_constraint=True))
    write(OUT/'preflight.json',dict(passed=True,checks=records))


if __name__=='__main__':main()
