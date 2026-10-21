"""Frozen contact formulas with nonfatal reporting of ambiguous source-side inputs.

Copied from pipeline_v2_contact_geometry/prepare.py. Only the negative source
skin-centroid assertion is changed to an audit record. Anchor, normal, patch,
gap, source masks and all downstream optimization/evaluation formulas are intact.
Open meshes do not provide a globally certified material sign.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import numpy as np
import trimesh
import coal
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from config import *
from geometry import object_patch,mesh_shape,unsigned_query

TIPS=np.array([4,8,12,16,20])

def audit_plot(seq,mesh,raw,local,pose,R,mask,closest,direction,patches,offsets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    active=np.flatnonzero(mask.any(axis=(1,2)))
    frames=active[np.array([0,len(active)//2,len(active)-1])]
    fig=plt.figure(figsize=(15,10))
    triangles=mesh.triangles;center_obj=triangles.mean(1)
    for row,side in enumerate(range(2)):
        for col,t in enumerate(frames):
            ax=fig.add_subplot(2,3,row*3+col+1,projection='3d')
            center=local[t,side,TIPS].mean(0);radius=.11
            vertices=(raw['vertices'][t]-pose[t,:3])@R[t]
            ht=vertices[raw['faces']]
            selected=np.linalg.norm(ht.mean(1)-center,axis=1)<radius
            ax.add_collection3d(Poly3DCollection(ht[selected],facecolor='#4c91bd',alpha=.35,linewidth=0))
            selected=np.linalg.norm(center_obj-center,axis=1)<radius*2
            ax.add_collection3d(Poly3DCollection(triangles[selected],facecolor='#b97842',alpha=.22,linewidth=0))
            for finger in range(5):
                if not mask[t,side,finger]:continue
                start,end=offsets[t,side,finger]
                ax.add_collection3d(Poly3DCollection(patches[start:end],facecolor='#44b56a',alpha=.6,linewidth=0))
                c=closest[t,side,finger];n=direction[t,side,finger]
                ax.scatter(*c,color='black',s=12);ax.text(*c,str(finger+1),fontsize=9)
                ax.quiver(*c,*n,length=.025,color='#bc2636',linewidth=1.3)
            ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),
                   zlim=(center[2]-radius,center[2]+radius),title=f'{("Left","Right")[side]} hand, frame {t}')
            ax.set_box_aspect((1,1,1));ax.view_init(elev=20,azim=45)
    fig.suptitle(seq+'\nBlue: source skin; orange: object; green: intended patches; red: exterior directions. Digits 1=thumb ... 5=little.')
    fig.tight_layout();fig.savefig(OUT/'targets'/f'{seq}_source_contacts.png',dpi=140);plt.close(fig)

def prepare(seq):
    info=inputs(seq);z=np.load(info['canonical'])
    with np.load(info['human']) as archive:raw={k:archive[k] for k in archive.files}
    human=z['hand_kp_scaled'];pose=z['object_pose_mujoco'];T=len(human)
    R=Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix()
    local=np.einsum('tij,thki->thkj',R,human-pose[:,None,None,:3])
    mesh=trimesh.load(info['object_mesh'],force='mesh');mesh.apply_scale(float(z['scale']))
    # Weld OBJ seam duplicates only for adjacency; triangle coordinates unchanged.
    before=mesh.triangles.copy();mesh.merge_vertices(merge_tex=True,merge_norm=True)
    assert np.allclose(before,mesh.triangles,rtol=0,atol=1e-9)
    assert mesh.is_winding_consistent
    closest,distance,face=trimesh.proximity.closest_point(mesh,local[:,:,TIPS].reshape(-1,3))
    closest=closest.reshape(T,2,5,3);distance=distance.reshape(T,2,5);face=face.reshape(T,2,5)
    mask=distance<.03
    assert np.array_equal(mask,np.load(info['masks'])['source_contact'])
    legacy_closest=closest.copy()
    legacy_normal=mesh.face_normals[face].copy();delta=local[:,:,TIPS]-closest
    normal=legacy_normal.copy()
    projection=np.sum(delta*normal,axis=-1)
    direction=np.where((distance>1e-6)[...,None],delta/np.maximum(distance[...,None],1e-9),normal)
    direction*=np.where(projection<0,-1.,1.)[...,None]
    # At a tangent/degenerate witness, use the locally oriented face normal.
    unstable=np.abs(projection)<1e-7
    direction[unstable]=normal[unstable]
    adjacency=[[] for _ in mesh.faces]
    for a,b in mesh.face_adjacency:adjacency[a].append(int(b));adjacency[b].append(int(a))
    patches=[];offsets=np.zeros((T,2,5,2),int);counts=np.zeros((T,2,5),int)
    source_gap=np.full((T,2,5),np.nan);roi_counts=np.zeros((T,2,5),int)
    source_side_ambiguities=[]
    skin_centroid=np.zeros((T,2,5,3));skin_centroid_plane=np.zeros((T,2,5))
    # Capture the actual human skin around each anatomical fingertip. No robot
    # output participates in these targets. Closest-tip assignment avoids
    # borrowing a neighboring digit when the fingertips are close together.
    for t in range(T):
        vertices=(raw['vertices'][t]-pose[t,:3])@R[t]
        ht=vertices[raw['faces']];centers=ht.mean(1)
        tips=local[t][:,TIPS,:].reshape(10,3)
        cd,owner=cKDTree(tips).query(centers)
        for side in range(2):
            for finger in range(5):
                if not mask[t,side,finger]:continue
                selected=np.flatnonzero((owner==side*5+finger)&(cd<EXTRA['human_tip_region_radius_m']))
                if len(selected)==0:
                    raise RuntimeError(f'No source fingertip skin at {seq} frame {t} digit {side,finger}')
                skin=ht[selected]
                area=np.linalg.norm(np.cross(skin[:,1]-skin[:,0],skin[:,2]-skin[:,0]),axis=-1)
                center=np.average(centers[selected],axis=0,weights=area)
                skin_centroid[t,side,finger]=center
                cp,cdist,cface=trimesh.proximity.closest_point(mesh,center[None])
                seed=int(cface[0]);face[t,side,finger]=seed
                # Select the face from the source finger skin, then project the
                # original landmark onto that face. This prevents a penetrated
                # landmark selecting the opposite side of a thin wall.
                anchor=trimesh.triangles.closest_point(mesh.triangles[[seed]],tips[[side*5+finger]])[0]
                closest[t,side,finger]=anchor;normal[t,side,finger]=mesh.face_normals[seed]
                outward=float((center-cp[0])@mesh.face_normals[seed])
                if outward < -.0001:
                    source_side_ambiguities.append(dict(frame=t,side=int(side),finger=int(finger),signed_plane_mm=float(outward*1000)))
                skin_centroid_plane[t,side,finger]=outward
                # Consistent local material orientation, with witness-vector
                # direction rather than a finger/pad orientation target.
                vec=center-cp[0]
                direction[t,side,finger]=(np.sign(outward)*vec/max(np.linalg.norm(vec),1e-9)
                    if abs(outward)>1e-7 else mesh.face_normals[seed])
                patch,ids=object_patch(mesh,seed,anchor,adjacency,
                    EXTRA['patch_radius_m'],EXTRA['patch_normal_cosine'])
                start=len(patches);patches.extend(patch);offsets[t,side,finger]=[start,len(patches)]
                roi_counts[t,side,finger]=len(selected);counts[t,side,finger]=len(patch)
                shape=mesh_shape(patch);human_shape=mesh_shape(ht[selected])
                gap,_,_=unsigned_query(human_shape,coal.Transform3s(),shape,coal.Transform3s())
                source_gap[t,side,finger]=0. if gap<EXTRA['source_zero_gap_m'] else gap
        if t%50==0:print('PREPARE',seq,t,T,flush=True)
    folder=OUT/'targets';folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/f'{seq}.npz',contact=mask,anchor_local=closest,legacy_anchor_local=legacy_closest,
        direction_local=direction,face_normal_local=normal,source_landmark_distance=distance,
        source_landmark_signed_plane=projection,source_skin_gap=source_gap,
        source_skin_centroid_local=skin_centroid,source_skin_centroid_plane=skin_centroid_plane,
        patch_triangles=np.asarray(patches),patch_offsets=offsets,patch_counts=counts,
        source_skin_triangle_counts=roi_counts,object_rotation=R,object_pose=pose)
    audit=dict(sequence=seq,frames=T,source_contact_finger_frames=int(mask.sum()),
        source_masks_unchanged=True,mesh_winding_consistent=True,mesh_watertight=bool(mesh.is_watertight),
        source_landmark_inside_local_face_gt1mm=int(np.sum(projection[mask]<-.001)),
        source_skin_selected_opposite_face_count=int(np.sum(np.sum(normal*legacy_normal,axis=-1)[mask]<0)),
        source_skin_centroid_inside_selected_face_gt1mm=int(np.sum(skin_centroid_plane[mask]<-.001)),
        new_anchor_shift_percentiles_mm=np.percentile(np.linalg.norm(closest-legacy_closest,axis=-1)[mask]*1000,[0,50,95,100]).tolist(),
        source_landmark_distance_percentiles_mm=np.percentile(distance[mask]*1000,[0,25,50,75,100]).tolist(),
        source_skin_gap_percentiles_mm=np.percentile(source_gap[mask]*1000,[0,25,50,75,100]).tolist(),
        source_skin_gap_under2mm_finger_frames=int(np.sum(source_gap[mask]<=.002)),
        patch_triangle_count_min_max=[int(counts[mask].min()),int(counts[mask].max())],
        source_skin_roi_triangle_count_min_max=[int(roi_counts[mask].min()),int(roi_counts[mask].max())],
        direction_degenerate_fallback_count=int(np.sum(np.abs(skin_centroid_plane[mask])<1e-7)),
        caveat='Source skin is reconstructed HUMOTO geometry, not tactile ground truth. Open mesh parts use consistent local face winding, not a global inside/outside SDF. Source skin intersection yields a zero unsigned target, never a penetration target.')
    audit.update(source_side_ambiguity_count=len(source_side_ambiguities),
        source_side_ambiguities=source_side_ambiguities,
        source_side_policy='Nonfatal input warning; retain existing source face/anchor/sign-corrected direction formula. No contact masking, frame exclusion, target replacement or optimization/evaluation change.')
    write(folder/f'{seq}.json',audit)
    audit_plot(seq,mesh,raw,local,pose,R,mask,closest,direction,np.asarray(patches),offsets)
    return audit
