"""Find a feasible first arm/hand pose before temporal warm starting.

Only the first frame is solved here. The caller retains its original subsequent
frame solver, contact targets, priors, body/object coordinates, and collision
geometry. Restoration may start infeasible; returned poses must pass nonlinear
collision checks. It never silently returns an infeasible initialization.
"""
from dataclasses import asdict, dataclass
import time
import numpy as np
from scipy import sparse
import clarabel
import mujoco
from scipy.spatial.transform import Rotation
from holosoma_retargeting.retarget_pipeline.clarabel_step import solve_clarabel_qp_step
from holosoma_retargeting.wuji.topoRetarget import wrist_frame


@dataclass(frozen=True)
class InitializationSettings:
    clearance_m: float = 0.
    restoration_iterations: int = 60
    refinement_iterations: int = 40
    fractions: tuple = tuple(2.**-i for i in range(9))
    positive_curl_seed_rad: float = .8
    pullback_offsets_m: tuple = (0.,.03,.06)
    palm_ik_iterations: int = 40
    palm_ik_rotation_length_m: float = .08
    feasibility_epsilon_m: float = 1e-7


def _violation(distance, clearance):
    v=np.maximum(clearance-distance,0.)
    return float(v@v),float(v.max(initial=0.))


def _qp(ref,q,r,J,d,G,radius,clearance,elastic):
    """Exact joint/trust bounds. Elastic feasibility restoration or hard refinement."""
    n,k=ref.n,len(d)
    H=2*(J.T@J+ref.cfg.damping*np.eye(n))
    if elastic:
        P=sparse.block_diag([sparse.csc_matrix(H),ref.cfg.collision_weight*sparse.eye(k)],format='csc')
        linear=np.r_[2*J.T@r,np.zeros(k)]
        I=sparse.hstack([sparse.eye(n),sparse.csc_matrix((n,k))],format='csc')
        C=sparse.hstack([-sparse.csc_matrix(G),-sparse.eye(k)],format='csc')
        Z=sparse.hstack([sparse.csc_matrix((k,n)),-sparse.eye(k)],format='csc')
        A=sparse.vstack([I,-I,C,Z],format='csc')
        b=np.r_[ref.hi-q[ref.qadr],q[ref.qadr]-ref.lo,d-clearance,np.zeros(k)]
    else:
        P=sparse.csc_matrix(H);linear=2*J.T@r;I=sparse.eye(n,format='csc')
        A=sparse.vstack([I,-I,-sparse.csc_matrix(G)],format='csc')
        b=np.r_[ref.hi-q[ref.qadr],q[ref.qadr]-ref.lo,d-clearance]
    cones=[clarabel.NonnegativeConeT(len(b)),clarabel.SecondOrderConeT(n+1)]
    A=sparse.vstack([A,sparse.csc_matrix((1,len(linear))),-I],format='csc')
    b=np.r_[b,radius,np.zeros(n)]
    x,audit=solve_clarabel_qp_step(P,linear,A,b,cones,ref.solver_settings)
    audit['slack_max_m']=float(x[n:].max(initial=0.)) if elastic else 0.
    return x[:n],audit


def _restore(ref,q,settings):
    """Prioritize actual aggregate penetration reduction, not task pose error."""
    q=q.copy();records=[]
    for it in range(settings.restoration_iterations):
        d,G,_=ref.collisions(q)
        violation,worst=_violation(d,settings.clearance_m)
        # Restore to near zero, leaving room inside the existing 1 mm tolerance.
        if d.min(initial=np.inf)>=-.5*ref.cfg.collision_tolerance:break
        chosen=None;radius=ref.cfg.trust_radius
        for shrink in range(6):
            dq,audit=_qp(ref,q,np.zeros(0),np.zeros((0,ref.n)),d,G,
                radius,settings.clearance_m,True)
            for alpha in settings.fractions:
                trial=q.copy();trial[ref.qadr]+=alpha*dq
                dd,_,_=ref.collisions(trial,False);value,peak=_violation(dd,settings.clearance_m)
                if value<violation-1e-14:
                    chosen=trial;break
            if chosen is not None:break
            radius*=.5
        records.append(dict(iteration=it,violation_squared_m2=violation,maximum_violation_m=worst,
            accepted=chosen is not None,alpha=alpha if chosen is not None else 0.,radius=radius,qp=audit))
        if chosen is None:break
        q=chosen
    d,_,_=ref.collisions(q,False)
    return q,records,bool(d.min(initial=np.inf)>=-ref.cfg.collision_tolerance+settings.feasibility_epsilon_m)


def _refine_feasible(ref,q,target,qbody,qinit,settings):
    """Original residuals/weights, first-frame hard collision acceptance only.

    Unlike a per-pair no-worsening guard, separated pairs can move freely up to
    the fixed tolerance. The guard cannot lock an initially penetrating pose:
    restoration must succeed before this phase is called.
    """
    q=q.copy();records=[]
    for it in range(settings.refinement_iterations):
        r,J=ref.residuals(q,target,qbody,qinit,None);d,G,_=ref.collisions(q)
        radius=ref.cfg.trust_radius;chosen=None;trials=[]
        for shrink in range(4):
            try:
                # A feasible current pose always admits dq=0 at this bound.
                dq,audit=_qp(ref,q,r,J,d,G,radius,-ref.cfg.collision_tolerance+2*settings.feasibility_epsilon_m,False)
            except RuntimeError as exc:
                trials.append(dict(radius=radius,error=str(exc)));radius*=.5;continue
            for alpha in settings.fractions:
                trial=q.copy();trial[ref.qadr]+=alpha*dq
                dd,_,_=ref.collisions(trial,False)
                feasible=bool(dd.min(initial=np.inf)>=-ref.cfg.collision_tolerance+settings.feasibility_epsilon_m)
                rr,_=ref.residuals(trial,target,qbody,qinit,None)
                value=float(rr@rr)
                trials.append(dict(radius=radius,alpha=alpha,cost=value,feasible=feasible,
                    max_depth_m=float(max(0.,-dd.min(initial=0.)))))
                if feasible and value<=float(r@r)+1e-10:
                    chosen=trial;break
            if chosen is not None:break
            radius*=.5
        records.append(dict(iteration=it,cost=float(r@r),accepted=chosen is not None,trials=trials,
            qp=audit if 'audit' in locals() else None))
        if chosen is None:break
        q=chosen
        if it>=2 and np.linalg.norm(alpha*dq)<ref.cfg.convergence_step:break
    return q,records


def _palm_seed(ref,q,target,offset,settings):
    """Place overlapping hands in the source palm orientation, outside the object.

    Pullback direction is object-surface-to-source-wrist, not a hard-coded world
    axis or a special cart coordinate. This is a seed, never a contact target.
    Collision restoration/validation is still required after this bounded IK.
    """
    q=q.copy();d,G,ids=ref.collisions(q)
    overlapping=[]
    for side in range(2):
        affected=np.linalg.norm(G[:,ref.side_columns[side]],axis=1)>1e-8
        if np.any(affected & (ref.pairs[ids,2]==0) & (d < -ref.cfg.collision_tolerance)):
            overlapping.append(side)
    for side in overlapping:
        ref.fk(q);p=ref.keypoints(False)[side]
        bid=int(ref.kb[side,0]);cols=ref.arm_columns[side];adr=ref.qadr[cols]
        semantic=wrist_frame(p,[1,5,9,13,17])
        orientation=target['palm'][side]@semantic.T@ref.d.xmat[bid].reshape(3,3)
        away=target['human'][side,0]-target['surface'][side].mean(axis=0)
        away/=max(float(np.linalg.norm(away)),1e-12)
        position=target['human'][side,0]+offset*away
        length=settings.palm_ik_rotation_length_m
        for _ in range(settings.palm_ik_iterations):
            ref.fk(q);point=ref.d.xpos[bid].copy();current=ref.d.xmat[bid].reshape(3,3)
            angular=Rotation.from_matrix(orientation@current.T).as_rotvec()
            err=np.r_[position-point,length*angular]
            if np.linalg.norm(position-point)<.0005 and np.linalg.norm(angular)<np.deg2rad(1):break
            jp=np.zeros((3,ref.m.nv));jr=np.zeros_like(jp)
            mujoco.mj_jac(ref.m,ref.d,jp,jr,point,bid)
            J=np.vstack([jp[:,ref.dadr[cols]],length*jr[:,ref.dadr[cols]]])
            step=np.linalg.solve(J.T@J+1e-4*np.eye(len(cols)),J.T@err)
            step*=min(1.,.2/max(float(np.abs(step).max()),1e-12))
            accepted=False
            for alpha in settings.fractions[:4]:
                trial=q.copy();trial[adr]=np.clip(q[adr]+alpha*step,ref.lo[cols],ref.hi[cols]);ref.fk(trial)
                test=np.r_[position-ref.d.xpos[bid],length*Rotation.from_matrix(orientation@ref.d.xmat[bid].reshape(3,3).T).as_rotvec()]
                if test@test<=err@err+1e-14:q=trial;accepted=True;break
            if not accepted or np.linalg.norm(alpha*step)<1e-5:break
    return q


def initialize_collision_aware(ref,target,qbody,qinit,human,settings=None):
    """Return a collision-checked frame 0; raise if no candidate is feasible.

    Two deterministic seeds: the existing quick fit and a positive-flexion
    quick fit. Both use the same original finger-prior target for final scoring.
    All 54 arm/finger coordinates may move; every other coordinate stays fixed.
    """
    settings=settings or InitializationSettings();started=time.perf_counter()
    cfg_before=vars(ref.cfg).copy()
    seeds={'original_quick_fit':qinit.copy()}
    seed=qbody.copy()
    for cols in ref.finger_columns:
        flex=cols.reshape(5,4)[1:,2:].ravel()
        seed[ref.qadr[flex]]=np.clip(settings.positive_curl_seed_rad,ref.lo[flex],ref.hi[flex])
    seeds['positive_curl_quick_fit']=ref.quick_fingers(seed[None],human[None])[0]
    for offset in settings.pullback_offsets_m:
        seeds[f'source_palm_pullback_{offset:g}m']=_palm_seed(ref,seeds['positive_curl_quick_fit'],target,offset,settings)
    candidates=[];audit=[]
    for name,seed in seeds.items():
        initial,_,_=ref.collisions(seed,False)
        q,restoration,ok=_restore(ref,seed,settings)
        record=dict(seed=name,initial_max_depth_m=float(max(0.,-initial.min(initial=0.))),
            restoration=restoration,restoration_feasible=ok)
        if ok:
            q,refinement=_refine_feasible(ref,q,target,qbody,qinit,settings)
            r,_=ref.residuals(q,target,qbody,qinit,None);d,_,_=ref.collisions(q,False)
            ok=bool(d.min(initial=np.inf)>=-ref.cfg.collision_tolerance+settings.feasibility_epsilon_m)
            record.update(refinement=refinement,final_feasible=ok,cost=float(r@r),
                final_max_depth_m=float(max(0.,-d.min(initial=0.))))
            if ok:candidates.append((float(r@r),name,q))
        audit.append(record)
    if not candidates:raise RuntimeError('No feasible first-frame arm/hand placement: '+repr([
        dict(seed=c['seed'],final_restoration=c['restoration'][-1]) for c in audit]))
    cost,name,q=min(candidates,key=lambda item:item[0])
    fixed=np.setdiff1d(np.arange(ref.m.nq),ref.qadr)
    np.testing.assert_array_equal(q[fixed],qbody[fixed])
    assert vars(ref.cfg)==cfg_before
    assert np.all(q[ref.qadr]>=ref.lo-1e-7) and np.all(q[ref.qadr]<=ref.hi+1e-7)
    return q,dict(selected_seed=name,cost=cost,seconds=time.perf_counter()-started,
        settings=asdict(settings),candidates=audit,body_object_unchanged=True,
        objective_weights_unchanged=True,original_finger_prior_target_unchanged=True,
        scope='First frame only: collision restoration followed by original task objective with no collision slack and nonlinear feasibility checks. Subsequent solver unchanged.')
