"""Physical tip-surface proximity and independent checks, separate from frozen metrics."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
import mujoco
import subprocess
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from config import *
from solver import ArmIK,SETTINGS


def main(variant_only):
    info=inputs();m=mujoco.MjModel.from_xml_path(str(info['model']));ref=ArmIK(m,info['object'])
    source=np.load(info['mask'])['source_contact'];selected=set(ref.joints)
    def chain(body):
        joints=[]
        while body:
            joints.extend(j for j in range(int(m.body_jntadr[body]),int(m.body_jntadr[body]+m.body_jntnum[body])) if j in selected)
            body=int(m.body_parentid[body])
        return tuple(joints[::-1])
    same_chain=np.array([chain(int(m.geom_bodyid[a]))==chain(int(m.geom_bodyid[b])) for a,b,_ in ref.pairs])
    def describe(g):return dict(geom_id=int(g),body=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])))
    def vertices(g):
        mid=int(m.geom_dataid[g]);s=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid])
        return m.mesh_vert[s:s+n].astype(float)@ref.d.geom_xmat[g].reshape(3,3).T+ref.d.geom_xpos[g]
    def certify(a,b):
        if any(m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH for g in (a,b)):return 'nonmesh_not_checked'
        eq=np.vstack([ConvexHull(vertices(g)).equations for g in (a,b)])
        result=linprog(np.zeros(3),A_ub=eq[:,:3],b_ub=-eq[:,3],bounds=[(None,None)]*3,method='highs',options={'threads':1})
        return 'convex_intersection_confirmed' if result.success else 'certified_separated' if result.status==2 else 'unresolved'
    output={}
    for variant in [variant_only]:
        dest=OUT/'variants'/variant/'clips'/SEQ;Q=np.load(dest/'final_v2.npz')['qpos']
        tip_distance=np.full(source.shape,.04);examples=[];misclassified=[];worst_fixed=None
        for t,q in enumerate(Q):
            d,G,ids,active,v=ref.geometry(q)
            bad=ids[(~active)&(v>1e-5)&(~same_chain[ids])]
            if len(bad):misclassified.append(dict(frame=t,pair_ids=bad.tolist()))
            for side in range(2):
                for f,g in enumerate(ref.tipgeom[side]):
                    mask=(ref.pairs[ids,0]==g)&(ref.pairs[ids,2]==0)
                    if mask.any():tip_distance[t,side,f]=d[mask].min()
            frozen_idx=np.flatnonzero((~active)&(v>1e-5))
            if len(frozen_idx):
                i=frozen_idx[np.argmax(v[frozen_idx])]
                if worst_fixed is None or v[i]>worst_fixed['violation_m']:
                    a,b,k=ref.pairs[ids[i]]
                    worst_fixed=dict(frame=t,pair_id=int(ids[i]),violation_m=float(v[i]),a=describe(a),b=describe(b),
                        same_active_joint_chain=bool(same_chain[ids[i]]),certificate=certify(a,b))
            # Certify each residual collision in IK; for the baseline check representative frames.
            chosen=np.flatnonzero(active&(d<-.001))
            if len(chosen) and (variant=='contact_ik' or t in (0,32,123,140,226)):
                i=chosen[np.argmin(d[chosen])];a,b,k=ref.pairs[ids[i]]
                examples.append(dict(frame=t,pair_id=int(ids[i]),depth_mm=float(-d[i]*1000),
                    a=describe(a),b=describe(b),certificate=certify(a,b)))
        stats={}
        for mm in (1,2,5):
            robot=(tip_distance<=mm/1000)&(tip_distance>=-SETTINGS['feasibility_tolerance'])
            tp=int(np.sum(robot&source));fp=int(np.sum(robot&~source));fn=int(np.sum(~robot&source))
            stats[f'tip_surface_recall_{mm}mm_pct']=100*tp/(tp+fn) if tp+fn else None
            stats[f'tip_surface_f1_{mm}mm_pct']=100*2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
        output[variant]=dict(**stats,worst_frozen_collision=worst_fixed,zero_gradient_but_nonrigid_pairs=misclassified,
            selected_penetration_certificates=examples,
            definitions='Each native robot fingertip collision convex against the object CoACD pieces using independent Coal queries; minimum signed gap per tip. Contact requires gap <= threshold and penetration <=10 micrometres. Same source per-tip contact mask as frozen metrics. This supplemental metric uses collision surfaces, not the fingertip-center distance used by the frozen evaluator. Distances capped at 40 mm.')
        np.savez_compressed(dest/'tip_collision_surface_distances.npz',tip_signed_distance_m=tip_distance,source_contact=source)
        write(dest/'physical_contact_audit.json',output[variant])
    return output


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=list(REMOVALS))
    args=parser.parse_args()
    if args.variant:main(args.variant)
    else:
        def worker(variant):
            subprocess.run([PY,__file__,'--variant',variant],check=True)
            return read(OUT/'variants'/variant/'clips'/SEQ/'physical_contact_audit.json')
        with ThreadPoolExecutor(max_workers=3) as pool:
            results=dict(zip(REMOVALS,pool.map(worker,REMOVALS)))
        write(OUT/'physical_contact_audit.json',results)
