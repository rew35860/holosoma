"""Separate moving-arm self collisions and independently audit large mesh distances.

Frozen evaluation files remain untouched. A positive projection gap certifies that
a reported negative mesh distance is a false positive. Intersections are checked
independently for the flagged large hand/object distances using convex halfspaces.
"""
import sys
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from config import *
sys.path.insert(0,str(RET/'retarget_pipeline'))
from pipeline_v2 import CoupledRefiner, Settings, structurally_adjacent


def main(variant,seq):
    dest=OUT/'variants'/variant/'clips'/seq;info=inputs(seq)
    m=mujoco.MjModel.from_xml_path(str(info['model']));ref=CoupledRefiner(m,info['object'],Settings(**settings(variant)));d=ref.d
    q=np.load(dest/'final_v2.npz')['qpos'];raw=np.load(dest/'per_frame_metrics.npz')
    moving_pairs={tuple(sorted((int(a),int(b)))) for a,b,k in ref.pairs if k==2}
    object_pairs={tuple(sorted((int(a),int(b)))) for a,b,k in ref.pairs if k==0}
    active=np.flatnonzero((m.geom_contype!=0)|(m.geom_conaffinity!=0))
    def name(kind,i):return mujoco.mj_id2name(m,kind,int(i)) or ''
    objects=[g for g in active if name(mujoco.mjtObj.mjOBJ_GEOM,g).startswith(info['object']+'_piece')]
    planes=[g for g in active if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE]
    robot=set(active)-set(objects)-set(planes)
    hands=[[g for g in robot if name(mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]).startswith(pre)] for pre in ('wj_left','wjr_right')]
    def describe(g):return dict(geom_id=int(g),geom=name(mujoco.mjtObj.mjOBJ_GEOM,g),body=name(mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]))
    def vertices(g):
        mid=int(m.geom_dataid[g]);s=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid])
        return m.mesh_vert[s:s+n].astype(float)@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
    def certificate(a,b,ft):
        if any(m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH for g in (a,b)):
            return dict(classification='unvalidated_nonmesh')
        va,vb=vertices(a),vertices(b)
        axes=list(np.eye(3));v=ft[:3]-ft[3:]
        if np.linalg.norm(v)>1e-9:axes.append(v/np.linalg.norm(v))
        gap=max(max((va@axis).min()-(vb@axis).max(),(vb@axis).min()-(va@axis).max()) for axis in axes)
        if gap>1e-7:return dict(classification='certified_separated_false_positive',projected_gap_mm=float(gap*1000))
        eq=np.vstack([ConvexHull(va).equations,ConvexHull(vb).equations])
        result=linprog(np.zeros(3),A_ub=eq[:,:3],b_ub=-eq[:,3],bounds=[(None,None)]*3,method='highs')
        return dict(classification='convex_intersection_confirmed_depth_not_certified' if result.success else 'convex_intersection_infeasible_false_positive' if result.status==2 else 'unresolved',lp_status=int(result.status))
    moving=np.zeros(len(q));fixed=np.zeros(len(q));records=[];self_records=[]
    # Raw all-frame hand max, with only audited >5 mm events recomputed; label this limited correction explicitly.
    screened=raw['hand_object_depth_m'].copy()
    for t in range(len(q)):
        d.qpos[:]=q[t];mujoco.mj_forward(m,d)
        for c in d.contact:
            a,b=int(c.geom1),int(c.geom2)
            if a not in robot or b not in robot or structurally_adjacent(m,int(m.geom_bodyid[a]),int(m.geom_bodyid[b])):continue
            depth=max(0.,-float(c.dist));is_moving=tuple(sorted((a,b))) in moving_pairs
            arr=moving if is_moving else fixed;arr[t]=max(arr[t],depth)
            if depth>.005:self_records.append(dict(frame=t,depth_mm=depth*1000,a=describe(a),b=describe(b),registered_moving_self_constraint=is_moving))
        for side in range(2):
            if raw['hand_object_depth_m'][t,side]<=.005:continue
            retained=0.
            for a in hands[side]:
                for b in objects:
                    if np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b])-m.geom_rbound[a]-m.geom_rbound[b]>.05:continue
                    ft=np.zeros(6);distance=float(mujoco.mj_geomDistance(m,d,int(a),int(b),.05,ft))
                    if distance>=0:continue
                    if distance<-.005:
                        cert=certificate(a,b,ft)
                        records.append(dict(frame=t,side=side,reported_depth_mm=-distance*1000,a=describe(a),b=describe(b),registered_object_constraint=tuple(sorted((a,b))) in object_pairs,**cert))
                        if 'false_positive' in cert['classification']:continue
                    retained=max(retained,-distance)
            screened[t,side]=retained
    def stats(a):return dict(mean_mm=float(a.mean()*1000),max_mm=float(a.max()*1000),duration_gt1mm_pct=float(100*np.mean(a>.001)),duration_gt5mm_pct=float(100*np.mean(a>.005)),duration_gt10mm_pct=float(100*np.mean(a>.01)))
    write(dest/'collision_audit.json',dict(sequence=seq,variant=variant,moving_arm_hand_self=stats(moving),other_robot_self=stats(fixed),
        moving_definition='Native MuJoCo contacts among the exact registered coupled-solver self pairs. Other contacts are generally unchanged lower-body geometry; records identify the pairs.',
        large_hand_object_events=records,large_self_events=self_records,
        screened_hand_object=stats(screened.max(1)),
        limitation='Only hand-object distances >5 mm independently screened. Small native depths and self-contact depths are retained; LP verifies intersection, not penetration magnitude. Original metrics unchanged.'))
    np.savez_compressed(dest/'collision_audit_per_frame.npz',moving_self_depth_m=moving,other_self_depth_m=fixed,screened_hand_depth_m=screened)


if __name__=='__main__':main(sys.argv[1],sys.argv[2])
