"""Check the added finger derivatives and unchanged arm/contact formulation."""
import numpy as np
import mujoco
from config import *
from solver import ArmIK, targets, SETTINGS


def main():
    info=inputs();q=np.load(info['baseline'])['qpos'];mask,surface=targets(info)
    model=mujoco.MjModel.from_xml_path(str(info['model']))
    ref=ArmIK(model,info['object'])
    oldmod=module('fixed_finger_solver_check',CODE.parent/'pipeline_contact_ik/solver.py')
    old=oldmod.ArmIK(model,info['object'])
    assert SETTINGS==oldmod.SETTINGS
    assert np.array_equal(ref.pairs,old.pairs) and np.array_equal(ref.margin,old.margin)
    assert np.array_equal(ref.qadr[ref.arms],old.qadr)
    rng=np.random.default_rng(96);eps=1e-6;checks=[]
    for t in (0,140,226):
        previous=q[t-1] if t else None
        r,A,s,B=ref.tasks(q[t],q[t],mask[t],surface[t],previous,previous)
        ro,Ao,so,Bo=old.tasks(q[t],q[t],mask[t],surface[t],previous,previous)
        assert np.allclose(r,ro,atol=1e-12)
        assert np.allclose(A[:,ref.arms],Ao,atol=1e-12)
        for kind,cols in (('arm',ref.arms),('finger',ref.fingers)):
            direction=np.zeros(ref.n);direction[cols]=rng.normal(size=len(cols))
            direction/=np.linalg.norm(direction)
            plus=q[t].copy();minus=q[t].copy()
            plus[ref.qadr]+=eps*direction;minus[ref.qadr]-=eps*direction
            rp,_,sp,_=ref.tasks(plus,q[t],mask[t],surface[t],previous,previous)
            rm,_,sm,_=ref.tasks(minus,q[t],mask[t],surface[t],previous,previous)
            pe=float(np.max(np.abs((rp-rm)/(2*eps)-A@direction)))
            se=float(np.max(np.abs((sp-sm)/(2*eps)-B@direction)))
            assert pe<1e-5 and se<1e-5,(t,kind,pe,se)
            checks.append(dict(frame=t,kind=kind,primary_error=pe,secondary_error=se))
        d,G,ids=ref.collisions(q[t]);scores=np.linalg.norm(G[:,ref.fingers],axis=1)
        # Test constraints with strong finger sensitivity, including negative distances.
        chosen=np.argsort(scores)[-5:]
        for i in chosen:
            direction=G[i]/np.linalg.norm(G[i]);plus=q[t].copy();minus=q[t].copy()
            plus[ref.qadr]+=eps*direction;minus[ref.qadr]-=eps*direction
            ga,gb,_=ref.pairs[ids[i]]
            ref.fk(plus);dp=ref.queries.query(ref.d,ga,gb)[0]
            ref.fk(minus);dm=ref.queries.query(ref.d,ga,gb)[0]
            error=float(abs((dp-dm)/(2*eps)-G[i]@direction))
            assert error<5e-5,(t,int(ids[i]),error)
            checks.append(dict(frame=t,pair_id=int(ids[i]),collision_derivative_error=error))
    write(OUT/'numerical_checks.json',dict(passed=True,active_dofs=ref.n,
        same_collision_pairs_margins_and_settings=True,same_primary_contact_and_arm_jacobian=True,
        source_mask_exact=True,checks=checks))
    print('Passed: 54-DOF task and collision Jacobians; unchanged targets, arm derivatives and settings.',flush=True)


if __name__=='__main__':main()
