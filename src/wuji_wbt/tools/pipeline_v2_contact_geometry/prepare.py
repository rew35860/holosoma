"""Freeze source-only contact targets, audit their geometry, record the protocol."""
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
                assert outward>=-.0001,(seq,t,side,finger,'Ambiguous source skin side',outward)
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
    write(folder/f'{seq}.json',audit)
    audit_plot(seq,mesh,raw,local,pose,R,mask,closest,direction,np.asarray(patches),offsets)
    return audit

def main():
    assert not (OUT/'protocol.json').exists(),'Do not overwrite an existing experiment'
    OUT.mkdir(parents=True,exist_ok=True)
    audits=[prepare(seq) for seq in COHORT]
    paths={str(p):sha(p) for seq in COHORT for p in inputs(seq).values() if isinstance(p,Path)}
    frozen=[RET/'retarget_pipeline/pipeline_v2.py',PILOT_CODE/'evaluate_one.py',
            CODE.parent/'ref_candidates_metrics/evaluate.py',CODE.parent/'pipeline_contact_ik/geometry.py',GUIDANCE/'render.py']
    sources=list(CODE.glob('*.py'))+frozen
    settings={seq:read(PILOT/'clips'/seq/'run.json')['settings'] for seq in COHORT}
    write(OUT/'protocol.json',dict(version='v2-contact-geometry-1',sequences=COHORT,
        variants={v:dict(label=LABELS[v],direction=flags[0],surface_gap=flags[1]) for v,flags in REMOVALS.items()},
        predeclared_before_optimization=True,branch='feature/wuji-pipeline-v2',settings=settings,extra=EXTRA,
        initialization='Reuse saved quick-finger initialization identically for all variants; original Omni body poses, 54 arm/finger DOFs. Root/waist/legs/object frozen exactly.',
        objective='E = E_original_V2 + direction_enabled * 2500 sum(r_side^2) + gap_enabled * 256 sum(r_gap^2). All original residuals, including point-to-anchor attraction, remain unchanged.',
        direction=dict(formula='r_side=min(0, n_source dot (p_geom_local(q)-source_anchor_local))',
            points='One existing geometry-reference point per fingertip; not a new pad point.',
            target='Select a local object face using the area-weighted centroid of source fingertip skin triangles, then project the original source fingertip landmark onto that face. Direction is the signed-corrected source skin-centroid-to-witness vector; fallback is the local triangle normal. This avoids a penetrated landmark picking the far face of a thin wall. These NEW targets are separate from unchanged legacy V2 attraction targets.',
            provenance='OTRetarget Eq.5 r_n, adapted to ten fingertip reference points.',
            differences='Local face orientation replaces the SDF sign where meshes are open; anatomical skin-region correspondence replaces optimal transport; binary existing 30-mm contact mask replaces distance kernels. A regional skin centroid is a side-selection heuristic, not a paper surface probe. No dense robot surface probes, link-count normalization, full-body/object optimization or geodesic location loss.'),
        gap=dict(formula='r_gap=max(0, distance(robot_tip_collision_convex, intended_original_surface_patch)) - source_skin_gap',
            source='Actual human skin triangles within 20 mm of the source fingertip landmark and assigned to that nearest fingertip. Minimum surface-to-surface distance to the same intended patch; below 0.1 mm clamped to zero. Positive approach/release gaps retained.',
            patch='Source anchor triangle plus connected triangles with normal dot seed_normal >=0.5, clipped to a +/-20 mm cube about the anchor. Fixed source-only per-frame patches; original mesh, not CoACD and not a convexified patch.',
            provenance='Inspired by OTRetarget separation residual Eq.5 r_d; this is an unsigned surface-pair adaptation, not its signed SDF probe residual.',
            differences='Uses existing robot tip collision convex and actual human skin ROI rather than OT-matched surface probes. Unsigned distance becomes zero at intersection; separate signed material-collision constraints remain essential. Existing V2 point-location objective retained; no added geodesic objective.'),
        dexweave='No new term is copied from DexWeave. Its upper-limb fingertip/orientation objectives and soft clearance motivate evaluating placement and collision separately; its clearance is not this target-gap attraction.',
        excluded=['Pad-facing preference','Fixed pad-point substitution','Force/wrench objective','Extra posture terms','Body/object reretargeting'],
        collision_control='All four newly run conditions use Coal GJK/EPA on the identical existing collision convexes (analytic plane). Pair filters, collision tolerance 1 mm, slack weight 1e5, activation 25 mm and original V2 nonlinear merit unchanged. Historical V2 used native MuJoCo and is not the matched control.',
        evaluation='Frozen pilot evaluator unchanged, including masks and thresholds. New direction/gap/Coal penetration and joint/smoothness diagnostics separately named, not substituted into historical results.',
        weights='2500 direction and 256 gap are initial un-tuned raw squared-metre coefficients inspired by OTRetarget. Without its kernels/probe normalization they do not reproduce its effective objective weighting.',
        resources=dict(workers=2,cpu_quota_percent=200,nice=19,memory_max_gib=5,memory_swap_max=0,gpu=False),
        limitations=['Two development clips, one run per condition.','Kinematic geometry does not prove load-bearing grasp stability.',
          'Legacy geometry-origin attraction can conflict with physical surface contact; retained to isolate additive terms.',
          'A zero whole-tip surface gap may involve an edge/back, because pad-facing is intentionally excluded.',
          'Convex collision approximation and local face winding require visual inspection.'],
        references=['https://arxiv.org/html/2609.36602v1#S3.SS3','https://arxiv.org/html/2609.34724v1#A1.SS3'],
        source_audits=audits,input_sha256=paths,source_sha256={str(p):sha(p) for p in sources},
        target_sha256={str(OUT/'targets'/f'{seq}.npz'):sha(OUT/'targets'/f'{seq}.npz') for seq in COHORT}))
    for p in sources:
        dest=OUT/'source_snapshot'/p.relative_to(WORK);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    write(OUT/'status.json',dict(state='prepared',completed=0,total=8))

if __name__=='__main__':main()
