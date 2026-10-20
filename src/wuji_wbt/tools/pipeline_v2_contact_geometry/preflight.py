"""Validate added analytic Jacobians and unchanged base residuals before solving."""
import numpy as np
from config import *
from solver import load,target_at,CoupledRefiner

def main():
    checks=[]
    for seq in COHORT:
        _,z,data,ref,body,init=load(seq,'both')
        Q=np.load(inputs(seq)['saved_v2'])['qpos']
        rng=np.random.default_rng(20261018)
        contact_frames=np.flatnonzero(data['contact'].any(axis=(1,2)))
        for t in [int(contact_frames[0]),int(contact_frames[len(contact_frames)//2]),int(contact_frames[-1])]:
            target=target_at(ref,z,data,t);q=Q[t].copy()
            r,J=ref.additional(q,target)
            # Validate arbitrary joint directions, both arm and finger columns.
            errors=[]
            for columns in [np.concatenate(ref.arm_columns),np.concatenate(ref.finger_columns)]:
                for _ in range(3):
                    v=np.zeros(ref.n);v[columns]=rng.normal(size=len(columns));v/=np.linalg.norm(v)
                    h=1e-6;plus=q.copy();minus=q.copy()
                    plus[ref.qadr]+=h*v;minus[ref.qadr]-=h*v
                    rp,_=ref.additional(plus,target);rm,_=ref.additional(minus,target)
                    numeric=(rp-rm)/(2*h);analytic=J@v
                    err=float(np.max(np.abs(numeric-analytic)))
                    errors.append(err)
            assert max(errors)<.005,(seq,t,errors)
            ref.use_gap=False;ref.use_direction=False
            a,A=ref.residuals(q,target,body[t],init[t],None)
            b,B=CoupledRefiner.residuals(ref,q,target,body[t],init[t],None)
            assert np.array_equal(a,b) and np.array_equal(A,B)
            ref.use_gap=True;ref.use_direction=True
            d,G,ids=ref.collisions(q)
            if len(d):
                v=rng.normal(size=ref.n);v/=np.linalg.norm(v);h=1e-6
                qp=q.copy();qm=q.copy();qp[ref.qadr]+=h*v;qm[ref.qadr]-=h*v
                dp,_,ip=ref.collisions(qp,False);dm,_,im=ref.collisions(qm,False)
                common=np.intersect1d(ids,np.intersect1d(ip,im))
                numeric=np.array([(dp[np.flatnonzero(ip==i)[0]]-dm[np.flatnonzero(im==i)[0]])/(2*h) for i in common])
                analytic=np.array([G[np.flatnonzero(ids==i)[0]]@v for i in common])
                collision_error=float(np.max(np.abs(numeric-analytic)))
                assert collision_error<.005,(seq,t,collision_error)
            checks.append(dict(sequence=seq,frame=t,max_added_jacobian_error=max(errors),
                collision_jacobian_error=collision_error,base_residual_exact=True))
    write(OUT/'preflight.json',dict(passed=True,checks=checks,epsilon=1e-6,tolerance=.005))

if __name__=='__main__':main()
