"""Frozen, CPU-only kinematic evaluation of saved reference stages; no optimization."""
from pathlib import Path
import os,sys,json,time,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import mujoco,trimesh
from scipy.spatial.transform import Rotation
WORK=Path(__file__).resolve().parents[4]
ROOT=WORK.parent
RET=WORK/'src/holosoma_retargeting/holosoma_retargeting'
BATCH=ROOT/'holosoma/eval_out/ref_candidates/batch_20261011_current_120'
OUT=BATCH/'evaluation_v1'
sys.path.insert(0,str(RET.parent))
from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS,JOINTS_MAPPINGS
SIDES=('left','right');PREFIX=('wj_left','wjr_right');TIPS=[4,8,12,16,20];MCP=[1,5,9,13,17];FPS=30.;DT=1/FPS
MAP=JOINTS_MAPPINGS['smplh_wuji_body','g1'];LANDMARKS=list(MAP)
BONES=[(s+'_'+a,s+'_'+b) for s in ('L','R') for a,b in [('Hip','Knee'),('Knee','Ankle'),('Ankle','Toe'),('Shoulder','Elbow'),('Elbow','Wrist')]]
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def mean(a):
 a=np.asarray(a);a=a[np.isfinite(a)];return float(a.mean()) if a.size else None
def percentile(a,q):
 a=np.asarray(a);a=a[np.isfinite(a)];return float(np.percentile(a,q)) if a.size else None
def name(m,kind,i):return mujoco.mj_id2name(m,kind,int(i)) or ''
def body(m,n):
 b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n);assert b>=0,n;return b
def joints(m):
 return {name(m,mujoco.mjtObj.mjOBJ_JOINT,j):(int(m.jnt_qposadr[j]),7 if m.jnt_type[j]==0 else 4 if m.jnt_type[j]==1 else 1) for j in range(m.njnt)}
def unit(a):
 norms=np.linalg.norm(a,axis=-1,keepdims=True);return a/np.maximum(norms,1e-12),norms[...,0]>1e-8
def angle(a,b):
 a,va=unit(a);b,vb=unit(b);v=np.degrees(np.arccos(np.clip(np.sum(a*b,axis=-1),-1,1)));return np.where(va&vb,v,np.nan)
def frames(kp):
 x,vx=unit(kp[...,9,:]-kp[...,0,:]);n,vn=unit(np.cross(x,kp[...,5,:]-kp[...,0,:]));y=np.cross(n,x);r=np.stack([x,y,n],axis=-1);r[~(vx&vn)]=np.nan;return r

def hand_ids(m):
 bids=[];tipgeoms=[]
 for pre in PREFIX:
  bids.append([body(m,pre+'_palm_link')]+[body(m,f'{pre}_finger{f}_link{l}') for f in range(1,6) for l in (1,3,4,4)])
  ids=[]
  for f in range(1,6):
   b=body(m,f'{pre}_finger{f}_link4');matches=[g for g in range(m.ngeom) if m.geom_bodyid[g]==b and m.geom_dataid[g]>=0 and f'finger{f}_tip_link' in name(m,mujoco.mjtObj.mjOBJ_MESH,m.geom_dataid[g]) and (m.geom_contype[g] or m.geom_conaffinity[g])]
   assert len(matches)==1,(pre,f,matches);ids.append(matches[0])
  tipgeoms.append(ids)
 return np.asarray(bids),np.asarray(tipgeoms)

def min_dist(m,d,aa,bb,cutoff=.05):
 """Exact native geometry signed distance within cutoff; bounding spheres prune only disjoint far pairs."""
 best=cutoff
 if not len(aa) or not len(bb):return best
 bb=np.asarray(bb)
 for a in aa:
  lower=np.linalg.norm(d.geom_xpos[bb]-d.geom_xpos[a],axis=1)-m.geom_rbound[bb]-m.geom_rbound[a]
  for ix in np.argsort(lower):
   if lower[ix]>max(best,0.):break
   best=min(best,float(mujoco.mj_geomDistance(m,d,int(a),int(bb[ix]),cutoff,None)))
 return best

def structurally_adjacent(m,a,b):
 # Exclude neighboring bodies connected through at most two articulated joints,
 # counting welded intermediate bodies as zero. Cross-limb pairs stay active.
 for child,ancestor in ((a,b),(b,a)):
  count=0
  while child>0 and count<=2:
   if child==ancestor:return True
   count+=int(m.body_jntnum[child]);child=int(m.body_parentid[child])
 return False

def extract(m,Q,obj,whole=True):
 T=len(Q);d=mujoco.MjData(m);hids,tids=hand_ids(m)
 active=np.flatnonzero((m.geom_contype!=0)|(m.geom_conaffinity!=0));og=np.asarray([g for g in active if name(m,mujoco.mjtObj.mjOBJ_GEOM,g).startswith(obj+'_piece')]);assert len(og)>0
 planes=[g for g in active if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE]
 hands=[[g for g in active if name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]).startswith(pre)] for pre in PREFIX];assert all(len(x)>0 for x in hands)
 handset=set(hands[0]+hands[1]);objset=set(og);pg=set(planes)
 nonhand=[g for g in active if g not in handset|objset|pg];rob=list(handset)+nonhand
 out=dict(hand_kp=np.empty((T,2,21,3)),hand_gap=np.full((T,2),.05),body_object_depth=np.zeros(T),ground_depth=np.zeros(T),self_depth=np.zeros(T))
 if whole:
  lids=[body(m,MAP[n]) for n in LANDMARKS];out['body_points']=np.empty((T,len(lids),3))
  sole=[[g for g in active if name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]).startswith(s+'_ankle_roll_sphere_')] for s in SIDES];assert all(len(v)==5 for v in sole)
  out['sole_height']=np.empty((T,2))
 for t in range(T):
  d.qpos[:]=Q[t];mujoco.mj_forward(m,d)
  out['hand_kp'][t]=d.xpos[hids];out['hand_kp'][t,:,TIPS,:]=d.geom_xpos[tids].transpose(1,0,2) if out['hand_kp'][t,:,TIPS,:].shape==(5,2,3) else d.geom_xpos[tids]
  for side in range(2):out['hand_gap'][t,side]=min_dist(m,d,hands[side],og)
  if whole:
   out['body_points'][t]=d.xpos[lids]
   out['body_object_depth'][t]=max(0.,-min_dist(m,d,nonhand,og))
   for g in rob:
    for plane in planes:out['ground_depth'][t]=max(out['ground_depth'][t],-float(mujoco.mj_geomDistance(m,d,int(g),int(plane),.05,None)))
   for side in range(2):out['sole_height'][t,side]=min(float(d.geom_xpos[g,2]-m.geom_size[g,0]) for g in sole[side])
   for c in d.contact:
    a,b=int(c.geom1),int(c.geom2)
    if a in rob and b in rob and not structurally_adjacent(m,int(m.geom_bodyid[a]),int(m.geom_bodyid[b])):out['self_depth'][t]=max(out['self_depth'][t],-float(c.dist))
 return out

def local(p,pose):
 R=Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix();return np.einsum('tij,t...i->t...j',R,p-pose[:,None,None,:3])
def nearest(pq,p):
 shape=p.shape;flat=p.reshape(-1,3);near=[];dist=[]
 for i in range(0,len(flat),16):
  a,b,_=pq.on_surface(flat[i:i+16]);near.append(a);dist.append(b)
 return np.concatenate(near).reshape(shape),np.concatenate(dist).reshape(shape[:-1])
def classification(src,pred):
 tp=int(np.count_nonzero(src&pred));fp=int(np.count_nonzero(~src&pred));fn=int(np.count_nonzero(src&~pred));den=2*tp+fp+fn
 return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/den if den else None)
def depth_stats(prefix,v,out):
 out[prefix+'_duration_gt1mm_pct']=float(np.mean(v>.001)*100);out[prefix+'_duration_gt10mm_pct']=float(np.mean(v>.01)*100);out[prefix+'_duration_gt20mm_pct']=float(np.mean(v>.02)*100)
 out[prefix+'_duration_gt5mm_pct']=float(np.mean(v>.005)*100)
 out[prefix+'_mean_mm']=float(v.mean()*1000);out[prefix+'_p95_mm']=float(np.percentile(v,95)*1000);out[prefix+'_max_mm']=float(v.max()*1000)

def metrics(x,source,pq,pose,floating):
 tips=x['hand_kp'][:,:,TIPS];tip_local=local(tips,pose);_,dist=nearest(pq,tip_local);human=source['hand_kp'][:,:,TIPS];hm=source['contact_hand'];contact=source['contact_tip'];contact_pred=dist<.03
 err=np.linalg.norm(tips-human,axis=-1);gap=np.linalg.norm(tip_local-source['surface_targets'],axis=-1)
 pn=angle(frames(source['hand_kp'])[...,:,2],frames(x['hand_kp'])[...,:,2]);out={}
 out['primary_tip_error_mm']=mean(err[:,:,:2][np.broadcast_to(hm[...,None],err[:,:,:2].shape)]*1000)
 out['secondary_tip_error_mm']=mean(err[:,:,2:][np.broadcast_to(hm[...,None],err[:,:,2:].shape)]*1000)
 out['palm_normal_error_deg']=mean(pn[hm]);out['palm_normal_invalid_contact_frames']=int(np.count_nonzero(hm&~np.isfinite(pn)))
 out['contact_location_gap_mm']=mean(gap[contact]*1000);out['contact_location_gap_p95_mm']=percentile(gap[contact]*1000,95)
 out['fingertip_surface_distance_on_contact_mm']=mean(dist[contact]*1000)
 scores={}
 for threshold in (.01,.02,.03):
  s=source['tip_distance']<threshold;p=dist<threshold;scores[str(int(threshold*1000))+'mm']=classification(s,p)
 c=scores['30mm'];out.update({f'contact_{k}_pct':v*100 if v is not None else None for k,v in c.items() if k in ('precision','recall','f1')})
 gated=classification(contact,contact_pred&(x['hand_gap'][...,None]>=-.001));out['contact_f1_penetration_gated_pct']=gated['f1']*100 if gated['f1'] is not None else None
 gate5=classification(contact,contact_pred&(x['hand_gap'][...,None]>=-.005));out['contact_f1_penetration_gated5mm_pct']=gate5['f1']*100 if gate5['f1'] is not None else None
 out['contact_f1_same_hand_pct']=classification(hm,contact_pred.any(-1))['f1'];out['contact_f1_same_hand_pct']=out['contact_f1_same_hand_pct']*100 if out['contact_f1_same_hand_pct'] is not None else None
 mask=contact[1:]&contact[:-1];slip=np.linalg.norm(np.diff(tip_local,axis=0)-np.diff(source['local_tips'],axis=0),axis=-1)/DT
 out['object_relative_fingertip_velocity_error_cm_s']=mean(slip[mask]*100)
 both_contact=hm[1:]&hm[:-1];srcR=np.einsum('tji,thjk->thik',source['object_rotation'],frames(source['hand_kp']));dstR=np.einsum('tji,thjk->thik',source['object_rotation'],frames(x['hand_kp']))
 src_inc=np.einsum('thji,thjk->thik',srcR[:-1],srcR[1:]);dst_inc=np.einsum('thji,thjk->thik',dstR[:-1],dstR[1:]);delta=np.einsum('thji,thjk->thik',src_inc,dst_inc)
 rot_error=np.degrees(np.arccos(np.clip((np.trace(delta,axis1=-2,axis2=-1)-1)/2,-1,1)));out['relative_hand_orientation_increment_error_deg']=mean(rot_error[both_contact])
 depths=np.maximum(0.,-x['hand_gap']);depth_stats('hand_object_pen',depths.max(axis=1),out)
 out['palm_to_floating_stageB_mm']=mean(np.linalg.norm(x['hand_kp'][:,:,0]-floating['hand_kp'][:,:,0],axis=-1)[hm]*1000)
 if 'body_points' in x:
  depth_stats('body_object_pen',x['body_object_depth'],out);depth_stats('robot_object_pen',np.maximum(depths.max(1),x['body_object_depth']),out);depth_stats('ground_pen',x['ground_depth'],out);depth_stats('self_pen_nonadjacent',x['self_depth'],out)
  out['any_pen_gt1mm_pct']=float(np.mean((depths.max(1)>.001)|(x['body_object_depth']>.001)|(x['ground_depth']>.001)|(x['self_depth']>.001))*100)
  out['any_pen_relaxed_thresholds_pct']=float(np.mean((depths.max(1)>.02)|(x['body_object_depth']>.02)|(x['ground_depth']>.01)|(x['self_depth']>.01))*100)
  robot=x['body_points'];human=source['body_points'];angles=[]
  for a,b in BONES:
   ia,ib=LANDMARKS.index(a),LANDMARKS.index(b);angles.append(angle(robot[:,ib]-robot[:,ia],human[:,ib]-human[:,ia]))
  out['body_bone_direction_error_deg']=mean(np.stack(angles,axis=1));out['body_joint_position_error_mm']=mean(np.linalg.norm(robot-human,axis=-1)*1000)
  rv=np.diff(robot,axis=0)/DT;hv=np.diff(human,axis=0)/DT;moving=np.linalg.norm(hv,axis=-1)>.02;va=angle(rv,hv);va[(np.linalg.norm(rv,axis=-1)<1e-8)&moving]=180.;out['link_velocity_direction_error_deg']=mean(va[moving]);out['body_jerk_mean_m_s3']=mean(np.linalg.norm(np.diff(robot,n=3,axis=0)/DT**3,axis=-1))
  toes=[LANDMARKS.index('L_Toe'),LANDMARKS.index('R_Toe')];speed=np.linalg.norm(np.diff(robot[:,toes,:2],axis=0),axis=-1)/DT;valid=source['support_mask'][1:]&source['support_mask'][:-1]
  eligible=valid.any(1);out['support_intervals']=int(eligible.sum());out['foot_skating_mean_cm_s']=mean(speed[valid]*100);out['foot_skating_max_m_s']=float(speed[valid].max()) if valid.any() else None
  for threshold in (.02,.30):out[f'foot_skating_duration_gt{int(threshold*100)}cm_s_pct']=float(np.mean(((speed>threshold)&valid).any(1)[eligible])*100) if eligible.any() else None
  support=source['support_mask'];out['support_foot_floating_gt3cm_pct']=mean((x['sole_height'][support]>.03).astype(float)*100)
  out['support_foot_height_mm']=mean(x['sole_height'][support]*1000)
 raw=dict(fingertip_distance=dist,contact_prediction=contact_pred,tip_error_m=err,contact_location_gap_m=gap,palm_error_deg=pn,hand_object_depth_m=depths)
 return out,dict(contact_counts=scores,penetration_gated_contact=gated),raw

def evaluate(seq):
 start=time.monotonic();job=read(BATCH/'jobs'/f'{seq}.json');assert job['status']=='success';obj=job['object'];clip=BATCH/'clips'/seq
 z=np.load(clip/'part0'/f'{seq}_canonical.npz',allow_pickle=True);Q=np.load(job['final_npz'])['qpos'];T=len(Q);pose=z['object_pose_mujoco'];scale=float(z['scale'])
 model=mujoco.MjModel.from_xml_path(job['model_xml']);am=mujoco.MjModel.from_xml_path(str(clip/'models'/seq/'stageA'/f'g1_29dof_wuji_welded_w_{obj}.xml'));hm=mujoco.MjModel.from_xml_path(str(clip/'models'/seq/'stageB'/f'hand_both_w_{obj}_coacd_ball.xml'))
 qa=np.load(clip/'part1'/f'{seq}_stageA.npz')['qpos'];qh=np.load(clip/'part2'/f'{seq}_stageB.npz')['qpos'];assert len(qa)==len(qh)==T==len(pose)
 jm,ja,jh=joints(model),joints(am),joints(hm);Qinit=np.tile(model.qpos0,(T,1));assert Q.shape==(T,83)
 for n,(a,w) in jm.items():
  source,mp=(qh,jh) if 'finger' in n and n in jh else (qa,ja)
  if n in mp:b,width=mp[n];assert width==w;Qinit[:,a:a+w]=source[:,b:b+w]
 Qinit[:,:7]=qa[:,:7];Qinit[:,-7:]=qa[:,-7:]
 arm=[a for n,(a,w) in jm.items() if any(n==f'{s}_{j}_joint' for s in SIDES for j in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw'))];assert len(arm)==14
 frozen=np.setdiff1d(np.arange(model.nq),arm);assert np.allclose(Q[:,frozen],Qinit[:,frozen],atol=1e-12,rtol=0);assert np.allclose(Q[:,-7:],pose,atol=1e-6,rtol=0)
 mesh=trimesh.load(RET/'models'/obj/f'{obj}.obj',force='mesh');mesh.apply_scale(scale);pq=trimesh.proximity.ProximityQuery(mesh)
 hk=z['hand_kp_scaled'];tip_local=local(hk[:,:,TIPS],pose);target,dist=nearest(pq,tip_local);human=z['human_joints_scaled'];hb=human[:,[SMPLH_DEMO_JOINTS.index(n) for n in LANDMARKS]]
 toes=hb[:,[LANDMARKS.index('L_Toe'),LANDMARKS.index('R_Toe')]];support=np.zeros((T,2),bool);support[1:]=np.linalg.norm(np.diff(toes[:,:,:2],axis=0),axis=-1)<=.01;support &= toes[:,:,2] <= np.min(toes[:,:,2])+.05
 source=dict(hand_kp=hk,body_points=hb,local_tips=tip_local,tip_distance=dist,surface_targets=target,contact_tip=dist<.03,contact_hand=(dist<.03).any(-1),support_mask=support,object_rotation=Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix())
 floating=extract(hm,qh,obj,False);variants={};counts={};raw={'source_tip_distance_m':dist,'source_contact':source['contact_tip'],'source_support':support}
 for label,extracted in [('floating_hands_stageB',floating),('body_plus_fingers_before_arm',extract(model,Qinit,obj)),('final_combined',extract(model,Q,obj))]:
  v,c,r=metrics(extracted,source,pq,pose,floating);variants[label]=v;counts[label]=c
  for k,val in r.items():raw[label+'__'+k]=val
  for k in ('body_object_depth','ground_depth','self_depth','sole_height'):
   if k in extracted:raw[label+'__'+k]=extracted[k]
 result=dict(evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),sequence=seq,object=obj,frames=T,scale=scale,variant_metrics=variants,counts=counts,source_contact_tip_frames=int(source['contact_tip'].sum()),source_contact_hand_frames=int(source['contact_hand'].sum()),frozen_body_and_fingers_verified=True,object_mesh_watertight=bool(mesh.is_watertight),evaluation_seconds=time.monotonic()-start,final_npz_sha256=hashlib.sha256(Path(job['final_npz']).read_bytes()).hexdigest())
 (OUT/'per_frame').mkdir(exist_ok=True);np.savez_compressed(OUT/'per_frame'/f'{seq}.npz',**raw);write(OUT/'per_sequence'/f'{seq}.json',result);print('EVALUATED',seq,round(result['evaluation_seconds'],1),'seconds',flush=True);return result
if __name__=='__main__':evaluate(sys.argv[1])
