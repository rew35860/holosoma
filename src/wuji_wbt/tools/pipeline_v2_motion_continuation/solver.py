"""Same physical objective and QP; vary only initialization/continuation.

Substeps interpolate prescribed transforms; object trajectory is never optimized.
Collision guards are an explicitly separate acceptance policy, not new losses.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import time
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from config import *
geometry = module('continuation_geometry_solver',GEOMETRY_CODE/'solver.py')
mujoco = geometry.mujoco
target_at = geometry.target_at
import pipeline_v2 as core

def R(q):
    return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()

def mix(a,b,u):
    if u==0.:return a.copy()
    if u==1.:return b.copy()
    q=(1-u)*a+u*b
    for sl in (slice(3,7),slice(-4,None)):
        Ra,Rb=R(a[sl]),R(b[sl])
        q[sl]=Rotation.from_matrix(Rotation.from_rotvec(u*Rotation.from_matrix(Rb@Ra.T).as_rotvec()).as_matrix()@Ra).as_quat()[[3,0,1,2]]
    return q

class Refiner(geometry.Refiner):
    def subproblem(self,q,residual,J,dist,gradient,previous):
        old=self.cfg.joint_rate_limit
        try:
            if getattr(self,'bound_previous',None) is not None:
                self.cfg.joint_rate_limit=old*self.bound_fraction
                previous=self.bound_previous
            return super().subproblem(q,residual,J,dist,gradient,previous)
        finally:self.cfg.joint_rate_limit=old

    def compensate(self,q,previous,current_contact,previous_contact,fraction):
        """Bounded palm-pose IK predictor; fingers unchanged; no added final loss."""
        result=q.copy();desired={}
        delta=R(q[-4:])@R(previous[-4:]).T
        self.fk(previous)
        for side in (0,1):
            if not (current_contact[side].any() and previous_contact[side].any()):continue
            bid=int(self.kb[side,0])
            desired[side]=(q[-7:-4]+delta@(self.d.xpos[bid]-previous[-7:-4]),delta@self.d.xmat[bid].reshape(3,3))
        length=OPTIONS['predictor_rotation_length_m'];steps=0
        for side,(position,orientation) in desired.items():
            cols=self.arm_columns[side];adr=self.qadr[cols];dadr=self.dadr[cols];bid=int(self.kb[side,0])
            lo=np.maximum(self.lo[cols],previous[adr]-self.cfg.joint_rate_limit*fraction)
            hi=np.minimum(self.hi[cols],previous[adr]+self.cfg.joint_rate_limit*fraction)
            for it in range(OPTIONS['predictor_iterations']):
                self.fk(result);p=self.d.xpos[bid].copy();r=self.d.xmat[bid].reshape(3,3).copy()
                err=np.r_[position-p,length*Rotation.from_matrix(orientation@r.T).as_rotvec()]
                jp=np.zeros((3,self.m.nv));jr=np.zeros_like(jp)
                mujoco.mj_jac(self.m,self.d,jp,jr,p,bid)
                J=np.vstack([jp[:,dadr],length*jr[:,dadr]])
                step=np.linalg.solve(J.T@J+OPTIONS['predictor_damping']*np.eye(7),J.T@err)
                step*=min(1.,OPTIONS['predictor_max_joint_step_rad']/max(np.abs(step).max(),1e-12))
                accepted=False
                for alpha in (1.,.5,.25,.125):
                    trial=result.copy();trial[adr]=np.clip(result[adr]+alpha*step,lo,hi)
                    self.fk(trial)
                    newerr=np.r_[position-self.d.xpos[bid],length*Rotation.from_matrix(orientation@self.d.xmat[bid].reshape(3,3).T).as_rotvec()]
                    if newerr@newerr<=err@err+1e-14:
                        result=trial;accepted=True;steps+=1;break
                if not accepted or np.linalg.norm(alpha*step)<1e-5:break
        assert np.array_equal(result[np.concatenate([self.qadr[c] for c in self.finger_columns])],q[np.concatenate([self.qadr[c] for c in self.finger_columns])])
        return result,dict(active_hands=list(desired),predictor_iterations=steps,
                           active_change_norm=float(np.linalg.norm(result[self.qadr]-q[self.qadr])))

    def solve_stage(self,q,target,qbody,qinit,previous,bound_previous=None,fraction=1.,first=False):
        self.bound_previous=bound_previous;self.bound_fraction=fraction
        limit=self.cfg.first_frame_iterations if first else self.cfg.max_iterations
        details=[];stopped=False
        for it in range(limit):
            residual,J=self.residuals(q,target,qbody,qinit,previous)
            distance,G,_=self.collisions(q)
            step,info=self.subproblem(q,residual,J,distance,G,previous)
            merit=float(residual@residual+.5*self.cfg.collision_weight*np.sum(np.maximum(-distance-self.cfg.collision_tolerance,0.)**2))
            chosen=None
            for alpha in (1.,.5,.25,.125):
                trial=q.copy();trial[self.qadr]+=alpha*step
                r,_=self.residuals(trial,target,qbody,qinit,previous)
                dd,_,_=self.collisions(trial,False)
                value=float(r@r+.5*self.cfg.collision_weight*np.sum(np.maximum(-dd-self.cfg.collision_tolerance,0.)**2))
                if value<=merit+1e-9:chosen=trial;break
            info.update(iteration=it,step_norm=float(np.linalg.norm(step)),merit=merit,accepted=chosen is not None,alpha=alpha if chosen is not None else 0.)
            details.append(info)
            if chosen is None:stopped=True;break
            q=chosen
            if it+1>=self.cfg.min_iterations and np.linalg.norm(alpha*step)<self.cfg.convergence_step:break
        self.bound_previous=None
        return q,dict(iterations=len(details),accepted=sum(s['accepted'] for s in details),linesearch_stopped=stopped,steps=details)

    def depth_map(self,q):
        d,_,ids=self.collisions(q,False)
        return {int(i):max(0.,-float(v)) for i,v in zip(ids,d)}

    def guard(self,q,reference,increase):
        actual=self.depth_map(q);tol=self.cfg.collision_tolerance
        worst=max((depth-max(tol,reference.get(i,0.))-increase for i,depth in actual.items()),default=-increase)
        return worst<=1e-7,float(max(0.,worst)),actual

    def intermediate_target(self,z,data,t,u,body_sub,endpoint):
        if u==1.:return endpoint
        rot=R(body_sub[-4:]);xyz=body_sub[-7:-4]
        local0=(z['hand_kp_scaled'][t-1]-z['object_pose_mujoco'][t-1,:3])@data['object_rotation'][t-1]
        local1=(z['hand_kp_scaled'][t]-z['object_pose_mujoco'][t,:3])@data['object_rotation'][t]
        human=((1-u)*local0+u*local1)@rot.T+xyz
        target=core.CoupledRefiner.frame_target(self,human,z['object_points_scaled'],body_sub[-7:],data['contact'][t],data['legacy_anchor_local'][t]@rot.T+xyz)
        target['frame']=t;target['patches']=endpoint['patches'];self.frame=t
        return target

    def continue_frame(self,z,data,t,body,init,previous,compensated):
        current=previous.copy();u=0.;attempts=[];accepted_count=0;all_steps=[];fallback=False
        endpoint=target_at(self,z,data,t)
        warm=body[t].copy();warm[self.qadr]=previous[self.qadr]
        self.fk(previous);prev_tip=(self.d.geom_xpos[self.tipgeom]-previous[-7:-4])@R(previous[-4:])
        prev_rot=R(previous[-4:]).T@self.d.xmat[int(self.kb[0,0])].reshape(3,3)
        self.fk(warm);cur_tip=(self.d.geom_xpos[self.tipgeom]-warm[-7:-4])@R(warm[-4:])
        cur_rot=R(warm[-4:]).T@self.d.xmat[int(self.kb[0,0])].reshape(3,3)
        rotation=float(Rotation.from_matrix(cur_rot@prev_rot.T).magnitude())
        distance=float(np.linalg.norm(cur_tip-prev_tip,axis=-1).max())
        nominal=min(1.,OPTIONS['continuation_tip_step_m']/max(distance,1e-12),OPTIONS['continuation_rotation_step_rad']/max(rotation,1e-12))
        # Adaptive retries are driven by actual signed collision depths below.
        step_fraction=max(OPTIONS['minimum_fraction'],nominal)
        while u<1.-1e-12:
            du=min(step_fraction,1.-u);v=min(1.,u+du)
            if len(attempts)>=OPTIONS['max_transition_attempts'] or accepted_count>=OPTIONS['max_accepted_substeps']:
                du=1.-u;v=1.;fallback=True
            sub_body=mix(body[t-1],body[t],v);sub_init=mix(init[t-1],init[t],v)
            target=self.intermediate_target(z,data,t,v,sub_body,endpoint)
            q=sub_body.copy();q[self.qadr]=current[self.qadr]
            predictor=None
            if compensated:q,predictor=self.compensate(q,current,data['contact'][t],data['contact'][t-1],du)
            reference=self.depth_map(current)
            initial_ok,initial_excess,_=self.guard(q,reference,OPTIONS['warm_depth_increase_m'])
            if not initial_ok and du>OPTIONS['minimum_fraction']+1e-12 and not fallback:
                attempts.append(dict(start=u,end=v,accepted=False,reason='initial_collision',excess_m=initial_excess,predictor=predictor))
                step_fraction=du/2;continue
            solved,record=self.solve_stage(q,target,sub_body,sub_init,previous,current,du)
            all_steps.extend(record['steps'])
            ok=True;excess=0.
            for alpha in OPTIONS['swept_check_fractions']:
                passed,value,_=self.guard(mix(current,solved,alpha),reference,OPTIONS['accepted_depth_increase_m'])
                ok &= passed;excess=max(excess,value)
            if not ok and du>OPTIONS['minimum_fraction']+1e-12 and not fallback:
                attempts.append(dict(start=u,end=v,accepted=False,reason='swept_collision',excess_m=excess,predictor=predictor,solve=record))
                step_fraction=du/2;continue
            unresolved=not (initial_ok and ok)
            attempts.append(dict(start=u,end=v,accepted=True,guard_unresolved=unresolved,resource_cap_fallback=fallback,
                                 warm_excess_m=initial_excess,swept_excess_m=excess,predictor=predictor,solve=record))
            current=solved;u=v;accepted_count+=1
            step_fraction=min(nominal,max(OPTIONS['minimum_fraction'],du*2))
        assert np.array_equal(current[np.setdiff1d(np.arange(self.m.nq),self.qadr)],body[t,np.setdiff1d(np.arange(self.m.nq),self.qadr)])
        assert np.abs(current[self.qadr]-previous[self.qadr]).max()<=self.cfg.joint_rate_limit+1e-6
        return current,dict(frame=t,iterations=len(all_steps),accepted=sum(s['accepted'] for s in all_steps),
            linesearch_stopped=any(a.get('solve',{}).get('linesearch_stopped',False) for a in attempts),steps=all_steps,
            attempts=attempts,substeps=accepted_count,guard_unresolved=sum(a.get('guard_unresolved',False) for a in attempts),
            nominal_tip_displacement_m=distance,resource_cap_fallback=fallback)


def load(seq,variant):
    result=geometry.load(seq,'geometry')
    result[3].__class__=Refiner
    result[3].experiment_variant='geometry'
    return result

def main(seq,variant):
    start=time.perf_counter();info,z,data,ref,body,init=load(seq,variant)
    dest=destination(seq,variant);dest.mkdir(parents=True,exist_ok=True)
    shutil.copy2(info['initialization'],dest/'finger_initialization.npz')
    ref.queries.log_path=dest/'collision_query_retries.json'
    records=[];Q=body.copy();previous=None
    solve_start=time.perf_counter()
    for t in range(len(Q)):
        if t and variant in ('substeps','combined'):
            q,record=ref.continue_frame(z,data,t,body,init,previous,variant=='combined')
        else:
            q=init[t].copy()
            if previous is not None:q[ref.qadr]=previous[ref.qadr]
            predictor=None
            if t and variant=='motion_init':q,predictor=ref.compensate(q,previous,data['contact'][t],data['contact'][t-1],1.)
            target=target_at(ref,z,data,t)
            q,record=ref.solve_stage(q,target,body[t],init[t],previous,first=t==0)
            record.update(frame=t,predictor=predictor,substeps=1,guard_unresolved=0,resource_cap_fallback=False)
        Q[t]=q;previous=q.copy();records.append(record)
        if t%10==0 or t==len(Q)-1:
            write(dest/'status.json',dict(state='running',frames=t+1,total=len(Q),elapsed_seconds=time.perf_counter()-start))
            print(seq,variant,t+1,len(Q),'substeps',record['substeps'],flush=True)
    elapsed=time.perf_counter()-solve_start
    fixed=np.setdiff1d(np.arange(ref.m.nq),ref.qadr)
    assert np.array_equal(Q[:,fixed],body[:,fixed]) and np.isfinite(Q).all()
    assert np.all(Q[:,ref.qadr]>=ref.lo-1e-7) and np.all(Q[:,ref.qadr]<=ref.hi+1e-7)
    assert np.abs(np.diff(Q[:,ref.qadr],axis=0)).max()<=ref.cfg.joint_rate_limit+1e-6
    if variant=='baseline':
        archived=np.load(PREVIOUS/'variants/geometry/clips'/seq/'final_v2.npz')['qpos']
        assert np.array_equal(Q,archived),'Control must reproduce archived geometry-side result exactly'
    np.savez_compressed(dest/'final_v2.npz',qpos=Q,fps=30,human_joints=z['human_joints_scaled'])
    write(dest/'iterations.json',records)
    write(dest/'run.json',dict(sequence=seq,variant=variant,frames=len(Q),settings=vars(ref.cfg),extra=EXTRA,
        signed_recovery=dict(enabled=False,weight=0.),geometry_side=True,hard_side_constraint=False,
        thumb_index_contact_multiplier=1,options=OPTIONS,joint_refinement_seconds=elapsed,total_seconds=time.perf_counter()-start,
        solver_seconds=sum(s['solver_seconds'] for r in records for s in r['steps']),
        line_search_stopped_frames=sum(r['linesearch_stopped'] for r in records),
        accepted_substeps=sum(r['substeps'] for r in records),guard_unresolved_substeps=sum(r['guard_unresolved'] for r in records),
        resource_cap_fallback_frames=sum(r['resource_cap_fallback'] for r in records),body_object_unchanged=True,
        control_matches_archived_exactly=variant=='baseline',final_sha256=sha(dest/'final_v2.npz')))
    write(dest/'status.json',dict(state='solved',frames=len(Q)))

if __name__=='__main__':main(*sys.argv[1:])
