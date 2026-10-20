"""14-DOF contact-first differential IK; fingers/body/object frozen per frame.

Priority: recover collision feasibility, minimize source contact-point residuals,
then choose posture in the primary task's linear nullspace. No interaction mesh
or reference-arm-angle objective is constructed. Failed feasibility is reported.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import time
import numpy as np
import mujoco
import clarabel
from scipy import sparse
from scipy.spatial.transform import Rotation
from config import *
sys.path.insert(0,str(RET/'retarget_pipeline'))
from pipeline_v2 import CoupledRefiner,Settings,TIPS,MCP,wrist_frame
from geometry import ConvexQueries

SETTINGS=dict(max_iterations=40,first_iterations=70,trust_radius=.25,rate_limit=.20,
    detect_distance=.04,torso_clearance=.005,self_clearance=0.,object_clearance=0.,ground_clearance=0.,
    feasibility_tolerance=1e-5,contact_scale=.01,contact_budget=.0005,
    posture_weight=.05,temporal_weight=.20,palm_weight=.05,step_damping=1e-5,
    recovery_slack_weight=1e8,collision_backend='Coal 3.0.2 convex GJK/EPA; analytic MuJoCo plane distance')


class ArmIK(CoupledRefiner):
    def __init__(self,m,obj):
        m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        super().__init__(m,obj,Settings(collision_detect=SETTINGS['detect_distance']))
        cols=np.concatenate(self.arm_columns)
        self.joints=np.asarray(self.joints)[cols]
        self.qadr=self.qadr[cols];self.dadr=self.dadr[cols]
        self.lo=self.lo[cols];self.hi=self.hi[cols];self.n=14
        self.arm_columns=[np.arange(7),np.arange(7,14)]
        self.middle=(self.lo+self.hi)/2;self.half=(self.hi-self.lo)/2
        def torso(g):
            name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or ''
            return any(k in name for k in ('torso','pelvis','waist'))
        self.torso=np.array([k==2 and (torso(a) or torso(b)) for a,b,k in self.pairs])
        self.margin=np.array([SETTINGS[('object_clearance','ground_clearance','self_clearance')[k]]
                              for _,_,k in self.pairs])
        self.margin[self.torso]=SETTINGS['torso_clearance']
        self.queries=ConvexQueries(m)

    def collisions(self,q,derivatives=True):
        self.fk(q)
        a,b=self.pairs[:,0],self.pairs[:,1]
        lower=np.linalg.norm(self.d.geom_xpos[a]-self.d.geom_xpos[b],axis=1)-self.m.geom_rbound[a]-self.m.geom_rbound[b]
        candidates=np.flatnonzero((lower<SETTINGS['detect_distance'])|self.is_plane)
        distances=[];gradients=[];ids=[]
        for i in candidates:
            ga,gb,_=self.pairs[i]
            distance,pa,pb,normal=self.queries.query(self.d,ga,gb,SETTINGS['detect_distance'])
            if distance>=SETTINGS['detect_distance']:continue
            distances.append(distance);ids.append(i)
            if derivatives:
                ja=np.zeros((3,self.m.nv));jb=np.zeros((3,self.m.nv))
                mujoco.mj_jac(self.m,self.d,ja,None,pa,int(self.m.geom_bodyid[ga]))
                mujoco.mj_jac(self.m,self.d,jb,None,pb,int(self.m.geom_bodyid[gb]))
                gradients.append(normal@(ja-jb)[:,self.dadr])
        return np.asarray(distances),np.asarray(gradients).reshape(-1,14),np.asarray(ids,int)

    def tasks(self,q,reference,contact,targets,previous,previous_ref):
        self.fk(q);p,J=self.keypoints();primary=[];jac=[]
        self.fk(reference);pref=self.keypoints(False);self.fk(q)
        for side in range(2):
            ids=np.asarray(TIPS)[contact[side]] if contact[side].any() else np.asarray(TIPS)
            target=targets[side,contact[side]] if contact[side].any() else pref[side,ids]
            primary.extend((p[side,ids]-target)/SETTINGS['contact_scale'])
            jac.extend(J[side,ids]/SETTINGS['contact_scale'])
        r=np.asarray(primary).reshape(-1);A=np.asarray(jac).reshape(-1,14)
        w=np.sqrt(SETTINGS['posture_weight'])
        secondary=[w*(q[self.qadr]-self.middle)/self.half];B=[w*np.diag(1/self.half)]
        if previous is not None:
            w=np.sqrt(SETTINGS['temporal_weight'])/.5
            secondary.append(w*((q-reference)-(previous-previous_ref))[self.qadr]);B.append(w*np.eye(14))
        w=np.sqrt(SETTINGS['palm_weight'])
        for side in range(2):
            current=wrist_frame(p[side],MCP);desired=wrist_frame(pref[side],MCP)
            jr=np.zeros((3,self.m.nv));mujoco.mj_jac(self.m,self.d,None,jr,p[side,0],int(self.kb[side,0]))
            for axis in (0,2):
                v=current[:,axis];skew=np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])
                secondary.append(w*(v-desired[:,axis]));B.append(-w*skew@jr[:,self.dadr])
        return r,A,np.concatenate(secondary),np.vstack(B)

    def geometry(self,q):
        d,G,ids=self.collisions(q)
        controllable=np.linalg.norm(G,axis=1)>1e-8
        violation=np.maximum(self.margin[ids]-d,0.)
        return d,G,ids,controllable,violation

    def qp(self,q,r,J,d,G,ids,previous,equality=None,recovery=False):
        n=14;k=len(d) if recovery else 0
        H=2*(J.T@J+SETTINGS['step_damping']*np.eye(n))
        P=sparse.block_diag([H,SETTINGS['recovery_slack_weight']*np.eye(k)],format='csc') if k else sparse.csc_matrix(H)
        linear=np.r_[2*J.T@r,np.zeros(k)]
        low=self.lo-q[self.qadr];high=self.hi-q[self.qadr]
        if previous is not None:
            low=np.maximum(low,previous[self.qadr]-SETTINGS['rate_limit']-q[self.qadr])
            high=np.minimum(high,previous[self.qadr]+SETTINGS['rate_limit']-q[self.qadr])
        blocks=[];rhs=[];cones=[]
        if equality is not None:
            E,target=equality
            blocks.append(sparse.hstack([sparse.csc_matrix(E),sparse.csc_matrix((len(E),k))]));rhs.append(target)
            cones.append(clarabel.ZeroConeT(len(E)))
        I=sparse.eye(n);Z=sparse.csc_matrix((n,k))
        inequalities=[sparse.hstack([I,Z]),sparse.hstack([-I,Z])]
        b=[high,-low]
        if len(d):
            inequalities.append(sparse.hstack([-sparse.csc_matrix(G),-sparse.eye(k) if k else sparse.csc_matrix((len(d),0))]))
            b.append(d-self.margin[ids])
        if k:
            inequalities.append(sparse.hstack([sparse.csc_matrix((k,n)),-sparse.eye(k)]));b.append(np.zeros(k))
        blocks.append(sparse.vstack(inequalities));rhs.append(np.concatenate(b));cones.append(clarabel.NonnegativeConeT(len(rhs[-1])))
        blocks.append(sparse.vstack([sparse.csc_matrix((1,n+k)),sparse.hstack([-I,Z])]))
        rhs.append(np.r_[SETTINGS['trust_radius'],np.zeros(n)]);cones.append(clarabel.SecondOrderConeT(n+1))
        problem=clarabel.DefaultSolver(sparse.triu(P).tocsc(),linear,sparse.vstack(blocks).tocsc(),np.concatenate(rhs),cones,self.solver_settings)
        result=problem.solve()
        ok=str(result.status) in ('Solved','AlmostSolved')
        return (np.asarray(result.x[:n]) if ok else None),dict(status=str(result.status),seconds=float(result.solve_time),
            slack_max_m=float(max(result.x[n:])) if ok and k else 0.)

    def frame(self,reference,contact,targets,previous,previous_ref):
        q=reference.copy()
        if previous is not None:
            # Transport the previous arm correction, then enforce the unchanged rate bound.
            q[self.qadr]+=previous[self.qadr]-previous_ref[self.qadr]
            q[self.qadr]=np.clip(q[self.qadr],np.maximum(self.lo,previous[self.qadr]-SETTINGS['rate_limit']),
                               np.minimum(self.hi,previous[self.qadr]+SETTINGS['rate_limit']))
        records=[];best_primary=np.inf;previous_sec=np.inf
        limit=SETTINGS['first_iterations'] if previous is None else SETTINGS['max_iterations']
        for it in range(limit):
            r,A,s,B=self.tasks(q,reference,contact,targets,previous,previous_ref)
            d,G,ids,active,v=self.geometry(q)
            vmax=float(v[active].max()) if active.any() else 0.
            recovery=vmax>SETTINGS['feasibility_tolerance']
            rms=float(np.linalg.norm(r)/np.sqrt(len(r)/3)*SETTINGS['contact_scale'])
            if not recovery:best_primary=min(best_primary,rms)
            if recovery:
                step,info=self.qp(q,np.zeros(14),np.eye(14)*.001,d[active],G[active],ids[active],previous,recovery=True)
                stage='feasibility_recovery'
            else:
                step,info=self.qp(q,r,A,d[active],G[active],ids[active],previous)
                stage='contact_then_nullspace_posture'
                if step is not None:
                    _,sing,V=np.linalg.svd(A,full_matrices=False)
                    rank=int(np.sum(sing>max(sing[0]*1e-7,1e-8)))
                    E=V[:rank]
                    second,detail=self.qp(q,s,B,d[active],G[active],ids[active],previous,equality=(E,E@step))
                    info['secondary']=detail
                    if second is not None:step=second
            if step is None:
                records.append(dict(iteration=it,stage=stage,accepted=False,**info));break
            accepted=False
            for alpha in (1.,.5,.25,.125,.0625,.03125,.015625):
                trial=q.copy();trial[self.qadr]+=alpha*step
                dd,gg,ii,aa,vv=self.geometry(trial)
                maximum=float(vv[aa].max()) if aa.any() else 0.
                rr,_,ss,_=self.tasks(trial,reference,contact,targets,previous,previous_ref)
                trial_rms=float(np.linalg.norm(rr)/np.sqrt(len(rr)/3)*SETTINGS['contact_scale'])
                if recovery:
                    accept=maximum<=SETTINGS['feasibility_tolerance'] or (np.sum(vv[aa]**2)<np.sum(v[active]**2)-1e-12 and maximum<=vmax+1e-6)
                else:
                    accept=maximum<=SETTINGS['feasibility_tolerance'] and (trial_rms<rms-1e-9 or
                        (trial_rms<=best_primary+SETTINGS['contact_budget'] and ss@ss<s@s-1e-9))
                if accept:q=trial;accepted=True;break
            records.append(dict(iteration=it,stage=stage,accepted=accepted,alpha=alpha if accepted else 0.,
                contact_rms_mm=rms*1000,controllable_violation_mm=vmax*1000,step_norm=float(np.linalg.norm(step)),**info))
            if not accepted or np.linalg.norm(alpha*step)<1e-5:break
        d,G,ids,active,v=self.geometry(q)
        stats=dict(iterations=len(records),steps=records,
            controllable_violation_mm=float(v[active].max()*1000) if active.any() else 0.,
            frozen_geometry_violation_mm=float(v[~active].max()*1000) if (~active).any() else 0.,
            frozen_violating_pair_ids=ids[(~active)&(v>SETTINGS['feasibility_tolerance'])].tolist(),
            violating_pair_ids=ids[active&(v>SETTINGS['feasibility_tolerance'])].tolist())
        stats['collision_feasible']=stats['controllable_violation_mm']<=SETTINGS['feasibility_tolerance']*1000
        return q,stats


def targets(info):
    import trimesh
    z=np.load(info['canonical']);human=z['hand_kp_scaled'];pose=z['object_pose_mujoco']
    mesh=trimesh.load(RET/'models'/info['object']/f"{info['object']}.obj",force='mesh',process=False)
    mesh.apply_scale(float(z['scale']));pq=trimesh.proximity.ProximityQuery(mesh)
    R=Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix()
    local=np.einsum('tij,thki->thkj',R,human[:,:,TIPS]-pose[:,None,None,:3])
    flat=local.reshape(-1,3);near=[];distance=[]
    for i in range(0,len(flat),16):
        a,b,_=pq.on_surface(flat[i:i+16]);near.append(a);distance.append(b)
    mask=np.concatenate(distance).reshape(local.shape[:-1])<.03
    surface=np.einsum('tij,thkj->thki',R,np.concatenate(near).reshape(local.shape))+pose[:,None,None,:3]
    assert np.array_equal(mask,np.load(info['mask'])['source_contact'])
    return mask,surface


def run(frames=None):
    started=time.perf_counter();info=inputs();Q=np.load(info['baseline'])['qpos'];contact,surface=targets(info)
    model=mujoco.MjModel.from_xml_path(str(info['model']));ref=ArmIK(model,info['object'])
    dest=OUT/('smoke' if frames else 'variants/contact_ik/clips')/SEQ;dest.mkdir(parents=True,exist_ok=True)
    reference=Q.copy();result=Q[:frames].copy() if frames else Q.copy();records=[];previous=None
    for t in range(len(result)):
        result[t],record=ref.frame(reference[t],contact[t],surface[t],previous,reference[t-1] if t else None)
        record['frame']=t;records.append(record);previous=result[t].copy()
        if t%10==0 or t==len(result)-1:
            write(dest/'status.json',dict(stage='solving',completed_frames=t+1,total_frames=len(result),
                feasible_frames=sum(r['collision_feasible'] for r in records),elapsed_seconds=time.perf_counter()-started))
            print('IK',t+1,len(result),'feasible',sum(r['collision_feasible'] for r in records),flush=True)
    frozen=np.setdiff1d(np.arange(model.nq),ref.qadr)
    assert np.array_equal(result[:,frozen],reference[:len(result),frozen])
    assert np.isfinite(result).all()
    np.savez_compressed(dest/'final_v2.npz',qpos=result,fps=30)
    np.savez_compressed(dest/'finger_initialization.npz',qpos=reference[:len(result)],fps=30)
    write(dest/'iterations.json',records)
    write(dest/'run.json',dict(sequence=SEQ,frames=len(result),active_dofs=14,settings=SETTINGS,
        frozen_body_fingers_object_exact=True,seconds=time.perf_counter()-started,
        feasible_frames=sum(r['collision_feasible'] for r in records),
        frozen_geometry_violation_frames=sum(r['frozen_geometry_violation_mm']>.01 for r in records),
        maximum_arm_step_rad=float(np.abs(np.diff(result[:,ref.qadr],axis=0)).max()) if len(result)>1 else 0.,
        pair_counts=ref.pair_categories,torso_pair_count=int(ref.torso.sum()),code_sha256=sha(__file__)))
    write(dest/'status.json',dict(stage='solved_pending_evaluation',frames=len(result)))
    return dest


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--frames',type=int);args=parser.parse_args()
    run(args.frames)
