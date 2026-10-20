"""Original V2 objective/solver plus independently switched side and gap losses."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import shutil
import time
import numpy as np
import mujoco
import coal
from config import *
from geometry import mesh_shape,unsigned_query
sys.path.insert(0,str(RET/'retarget_pipeline'))
from pipeline_v2 import CoupledRefiner,Settings,TIPS,initial_body
CQ=module('contact_geometry_coal',CODE.parent/'pipeline_contact_ik/geometry.py').ConvexQueries

class CheckedQueries(CQ):
    """Reject Coal failure sentinels; retry signed convex queries deterministically."""
    def __init__(self,m):
        super().__init__(m);self.events=[];self.log_path=None

    def query(self,d,a,b,cutoff=.025):
        a,b=int(a),int(b)
        def valid(result):
            value,pa,pb,n=result
            if not np.isfinite(value) or not np.isfinite(np.r_[pa,pb,n]).all():return False
            if self.shapes[a] is None or self.shapes[b] is None:return True
            radius=float(self.m.geom_rbound[a]+self.m.geom_rbound[b])
            separation=float(np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b]))
            return (-2*radius-1e-6<=value<=separation+radius+1e-6 and np.linalg.norm(n)>.5)
        try:
            original=super().query(d,a,b,cutoff)
            if valid(original):return original
            failure=repr(original[0])
        except (AssertionError,RuntimeError) as exc:failure=repr(exc)
        Ta=coal.Transform3s(d.geom_xmat[a].reshape(3,3),d.geom_xpos[a])
        Tb=coal.Transform3s(d.geom_xmat[b].reshape(3,3),d.geom_xpos[b])
        candidates=[]
        for reverse in (False,True):
            req=coal.DistanceRequest(True,0.,0.);req.enable_signed_distance=True
            req.gjk_tolerance=1e-8;req.epa_tolerance=1e-7
            req.gjk_max_iterations=1024;req.epa_max_iterations=1024
            result=coal.DistanceResult()
            if reverse:
                value=float(coal.distance(self.shapes[b],Tb,self.shapes[a],Ta,req,result))
                candidate=(value,np.asarray(result.getNearestPoint2()).copy(),np.asarray(result.getNearestPoint1()).copy(),np.asarray(result.normal).copy())
            else:
                value=float(coal.distance(self.shapes[a],Ta,self.shapes[b],Tb,req,result))
                candidate=(value,np.asarray(result.getNearestPoint1()).copy(),np.asarray(result.getNearestPoint2()).copy(),-np.asarray(result.normal).copy())
            if valid(candidate):candidates.append(candidate)
        if not candidates:
            raise RuntimeError(f'All signed convex distance retries failed for geom pair {a,b}: {failure}')
        if len(candidates)==2 and abs(candidates[0][0]-candidates[1][0])>1e-4:
            raise RuntimeError(f'Signed distance retries disagree for {a,b}: {[x[0] for x in candidates]}')
        chosen=candidates[0]
        self.events.append(dict(pair=[a,b],original_failure=failure,retry_distances_m=[x[0] for x in candidates],qpos=d.qpos.tolist()))
        if self.log_path:write(self.log_path,dict(events=self.events))
        return chosen

class Refiner(CoupledRefiner):
    def __init__(self,m,obj,settings,data,variant):
        super().__init__(m,obj,settings)
        self.queries=CheckedQueries(m);self.data=data;self.variant=variant;self.frame=-1
        self.use_direction,self.use_gap=REMOVALS[variant]

    def collisions(self,q,derivatives=True):
        self.fk(q);a,b=self.pairs[:,:2].T
        lower=np.linalg.norm(self.d.geom_xpos[a]-self.d.geom_xpos[b],axis=1)-self.m.geom_rbound[a]-self.m.geom_rbound[b]
        candidates=np.flatnonzero((lower<self.cfg.collision_detect)|self.is_plane)
        distances=[];gradients=[];ids=[]
        for i in candidates:
            ga,gb,_=self.pairs[i]
            dist,pa,pb,normal=self.queries.query(self.d,ga,gb,self.cfg.collision_detect)
            if dist>=self.cfg.collision_detect:continue
            distances.append(dist);ids.append(i)
            if derivatives:
                ja=np.zeros((3,self.m.nv));jb=np.zeros((3,self.m.nv))
                mujoco.mj_jac(self.m,self.d,ja,None,pa,int(self.m.geom_bodyid[ga]))
                mujoco.mj_jac(self.m,self.d,jb,None,pb,int(self.m.geom_bodyid[gb]))
                gradients.append(normal@(ja-jb)[:,self.dadr])
        return np.asarray(distances),np.asarray(gradients).reshape(-1,self.n),np.asarray(ids,int)

    def frame_target(self,*args):
        target=super().frame_target(*args);self.frame+=1;t=self.frame
        target['frame']=t;target['patches']={}
        if self.use_gap:
            self.add_patches(target)
        return target

    def add_patches(self,target):
        t=target['frame']
        for side,finger in np.argwhere(self.data['contact'][t]):
            start,end=self.data['patch_offsets'][t,side,finger]
            target['patches'][int(side),int(finger)]=mesh_shape(self.data['patch_triangles'][start:end])

    def gap_query(self,target,side,finger,derivatives=True):
        g=int(self.tipgeom[side,finger]);R=target['R'];translation=target['object_pose'][:3]
        transform=coal.Transform3s(self.d.geom_xmat[g].reshape(3,3),self.d.geom_xpos[g])
        gap,pa,pb=unsigned_query(self.queries.shapes[g],transform,
            target['patches'][side,finger],coal.Transform3s(R,translation))
        grad=np.zeros(self.n)
        if derivatives and gap>1e-8:
            jp=np.zeros((3,self.m.nv))
            mujoco.mj_jac(self.m,self.d,jp,None,pa,int(self.m.geom_bodyid[g]))
            delta=pa-pb;grad=(delta/max(np.linalg.norm(delta),1e-12))@jp[:,self.dadr]
        return gap,grad,pa,pb

    def additional(self,q,target,force_all=False):
        self.fk(q);points,jac=self.keypoints();t=target['frame'];r=[];J=[]
        if (self.use_gap or force_all) and not target['patches']:self.add_patches(target)
        for side,finger in np.argwhere(self.data['contact'][t]):
            side,finger=int(side),int(finger);tip=TIPS[finger]
            if self.use_direction or force_all:
                n=target['R']@self.data['direction_local'][t,side,finger]
                anchor=target['R']@self.data['anchor_local'][t,side,finger]+target['object_pose'][:3]
                signed=float(n@(points[side,tip]-anchor));w=np.sqrt(EXTRA['direction_weight'])
                r.append(w*min(signed,0.));J.append(w*n@jac[side,tip] if signed<0 else np.zeros(self.n))
            if self.use_gap or force_all:
                gap,gradient,_,_=self.gap_query(target,side,finger)
                w=np.sqrt(EXTRA['gap_weight'])
                r.append(w*(gap-self.data['source_skin_gap'][t,side,finger]));J.append(w*gradient)
        return np.asarray(r),np.asarray(J).reshape(-1,self.n)

    def residuals(self,q,target,q_body,q_finger,previous):
        r,J=super().residuals(q,target,q_body,q_finger,previous)
        if not (self.use_direction or self.use_gap):return r,J
        extra,G=self.additional(q,target)
        return np.r_[r,extra],np.vstack([J,G])

def load(seq,variant):
    info=inputs(seq)
    with np.load(info['canonical']) as archive:z={k:archive[k] for k in archive.files}
    with np.load(OUT/'targets'/f'{seq}.npz') as archive:data={k:archive[k] for k in archive.files}
    m=mujoco.MjModel.from_xml_path(str(info['model']))
    settings=Settings(**read(PILOT/'clips'/seq/'run.json')['settings'])
    ref=Refiner(m,info['object'],settings,data,variant)
    ma=mujoco.MjModel.from_xml_path(str(info['body_model']))
    body=initial_body(m,ma,np.load(info['body'])['qpos'])
    init=np.load(info['initialization'])['qpos']
    return info,z,data,ref,body,init

def target_at(ref,z,data,t):
    ref.frame=t-1
    R=data['object_rotation'][t]
    surface=data['legacy_anchor_local'][t]@R.T+data['object_pose'][t,:3]
    return ref.frame_target(z['hand_kp_scaled'][t],z['object_points_scaled'],
        z['object_pose_mujoco'][t],data['contact'][t],surface)

def main(seq,variant):
    start=time.perf_counter();info,z,data,ref,body,init=load(seq,variant)
    dest=destination(seq,variant);dest.mkdir(parents=True,exist_ok=True)
    ref.queries.log_path=dest/'collision_query_retries.json'
    shutil.copy2(info['initialization'],dest/'finger_initialization.npz')
    surface=np.einsum('tij,thkj->thki',data['object_rotation'],data['legacy_anchor_local'])+data['object_pose'][:,None,None,:3]
    preparation=time.perf_counter()-start
    def progress(done,total,record):
        write(dest/'status.json',dict(state='running',frames=done,total=total,elapsed_seconds=time.perf_counter()-start))
        print(variant,seq,done,total,flush=True)
    solve_start=time.perf_counter()
    Q,records=ref.refine(body,init,z['hand_kp_scaled'],z['object_points_scaled'],z['object_pose_mujoco'],data['contact'],surface,progress)
    elapsed=time.perf_counter()-solve_start
    np.savez_compressed(dest/'final_v2.npz',qpos=Q,fps=30,human_joints=z['human_joints_scaled'])
    write(dest/'iterations.json',records)
    write(dest/'run.json',dict(sequence=seq,variant=variant,frames=len(Q),settings=vars(ref.cfg),
        extra=EXTRA,changed_residuals=REMOVALS[variant],initialization_reused=True,
        preparation_seconds=preparation,joint_refinement_seconds=elapsed,
        total_seconds=time.perf_counter()-start,active_dofs=ref.n,collision_pair_counts=ref.pair_categories,
        collision_backend='Coal convex signed distance; original V2 tolerances and constraints',
        signed_query_retry_count=len(ref.queries.events),
        line_search_stopped_frames=sum(r['linesearch_stopped'] for r in records),
        maximum_step_slack_m=max(s['slack_max_m'] for r in records for s in r['steps']),
        solver_seconds=sum(s['solver_seconds'] for r in records for s in r['steps']),
        final_sha256=sha(dest/'final_v2.npz')))
    write(dest/'status.json',dict(state='solved',frames=len(Q)))

if __name__=='__main__':main(*sys.argv[1:])
