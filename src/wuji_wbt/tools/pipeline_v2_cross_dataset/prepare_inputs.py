"""Source adapters, isolated colliders and model construction, without GPU use."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import time,shutil,fcntl,xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from config import *
sys.path.insert(0,str(RET.parent))
from holosoma_retargeting.retarget_pipeline.pipeline import PipelineConfig,build_scaled_model
from holosoma_retargeting.retarget_pipeline.object_colliders import validate_body_object_colliders

def make_geometry(r):
    obj=r['object'];folder=OUT/'geometry'/obj;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'build.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (folder/'build.json').exists():return
        start=time.perf_counter()
        source=Path(r['metadata']['mesh']) if r['dataset']=='hiphi' else RET/'models'/obj/f'{obj}.obj'
        mesh=trimesh.load(source,force='mesh');mesh.apply_scale(.01 if r['dataset']=='hiphi' else 1.)
        original_triangles=mesh.triangles.copy()
        mesh.merge_vertices(merge_tex=True,merge_norm=True)
        np.testing.assert_allclose(mesh.triangles,original_triangles,rtol=0,atol=1e-9)
        winding_repaired=not mesh.is_winding_consistent
        if winding_repaired:
            # Change only triangle orientation, never vertices/thickness/openings.
            original_faces=np.sort(mesh.faces,axis=1).copy()
            trimesh.repair.fix_normals(mesh,multibody=True)
            np.testing.assert_array_equal(np.sort(mesh.faces,axis=1),original_faces)
            assert mesh.is_winding_consistent
        mesh.export(folder/'visual.obj')
        components=ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_proxy_motion_20261021/geometry/components'/obj
        if obj in ('utility_cart','step_stool'):
            files=sorted(components.glob('piece_*.obj'));assert files
            for p in files:shutil.copy2(p,folder/p.name)
            method='Previously validated object-specific components';parameters=read(components/'build.json')
        else:
            from convex_components import candidate
            hulls,component_audit=candidate(mesh)
            write(folder/'component_candidate.json',component_audit)
            if hulls:
                for i,hull in enumerate(hulls):hull.export(folder/f'piece_{i:03d}.obj')
                method='Automatic convex connected components; 0.5mm bidirectional sampled surface budget'
                parameters=dict(max_pieces=32,budget_m=.0005,component_audit=component_audit)
            else:
                import coacd
                budget=module('cross_budget',CODE.parent/'pipeline_v2_proxy_budget/build.py')
                parameters=budget.PARAMS.copy();coacd.set_log_level('warn')
                parts=coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices,np.float64),np.asarray(mesh.faces,np.int32)),**parameters)
                assert 0<len(parts)<=32
                for i,(v,f) in enumerate(parts):trimesh.Trimesh(v,f,process=False).export(folder/f'piece_{i:03d}.obj')
                method='UMR/GRAIL parameter preset; current CoACD library'
        write(folder/'build.json',dict(object=obj,method=method,parameters=parameters,
            seconds=time.perf_counter()-start,pieces=len(list(folder.glob('piece_*.obj'))),
            source=str(source),source_sha256=sha(source),visual_sha256=sha(folder/'visual.obj'),
            object_local_mesh_preserved=True,winding_repaired=winding_repaired,
            winding_policy='Repair inconsistent adjacent triangle winding only; vertices and triangle sets unchanged. Open surfaces still lack a globally certified inside/outside sign.'))

def humoto_source(r):
    seq=r['sequence'];source=Path(r['source']);old=source/'clips'/seq/'part0'/f'{seq}_canonical.npz'
    with np.load(old) as f:z=dict(f)
    count=min(r['maximum_frames'],len(z['hand_kp_scaled']));total=len(z['hand_kp_scaled'])
    for k,v in list(z.items()):
        if v.ndim and len(v)==total and k!='object_points_scaled':z[k]=v[:count]
    skin=NATIVE/'human'/f'{seq}.npz'
    if not skin.exists():
        prep=module('cross_humoto_skin',PILOT_CODE/'prepare_human.py');prep.OUT=OUT
        skin=prep.prepare(seq)
    with np.load(skin) as f:human=dict(f)
    human['vertices']=human['vertices'][:count]
    if source!=NATIVE:
        shift=np.array([0.,0.,float(z['scale']*z['z_min'])])
        z['human_joints_scaled']=z['human_joints_scaled']+shift
        z['hand_kp_scaled']=z['hand_kp_scaled']+shift
        z['object_pose_mujoco']=z['object_pose_mujoco'].copy();z['object_pose_mujoco'][:,:3]+=shift
        human['vertices']=human['vertices']+shift
    z['floor_mode']=np.array('source');z['frame_ids']=np.arange(count)
    return z,human

def hiphi_source(r):
    import torch,smplx,csv
    from smplx.vertex_ids import vertex_ids
    seq=r['sequence'];fit=np.load(r['source']);T=min(r['maximum_frames'],len(fit['poses']))
    model=smplx.SMPLX(str(ROOT/'hiphi2smplx/assets/models/SMPLX_NEUTRAL.npz'),use_pca=False,
                      flat_hand_mean=True,batch_size=16)
    pose=fit['poses'][:T];trans=fit['trans'][:T];verts=[];joints=[]
    with torch.no_grad():
        for i in range(0,T,16):
            n=min(16,T-i);p=np.pad(pose[i:i+n],((0,16-n),(0,0)),mode='edge')
            t=np.pad(trans[i:i+n],((0,16-n),(0,0)),mode='edge');p=torch.from_numpy(p)
            out=model(global_orient=p[:,:3],body_pose=p[:,3:66],jaw_pose=p[:,66:69],
                leye_pose=p[:,69:72],reye_pose=p[:,72:75],left_hand_pose=p[:,75:120],right_hand_pose=p[:,120:165],
                betas=torch.from_numpy(np.repeat(fit['betas'][None],16,axis=0)),transl=torch.from_numpy(t),return_verts=True)
            verts.append(out.vertices[:n].numpy().copy());joints.append(out.joints[:n].numpy().copy())
    A=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
    metadata=read(Path(r['metadata']['clip_folder'])/'metadata.json');scale=1.32/(metadata['actor_metadata']['height_cm']/100)
    vertices=np.concatenate(verts)@A.T*scale;j=np.concatenate(joints)@A.T*scale
    hk=np.zeros((T,2,21,3));digits=('thumb','index','middle','ring','pinky')
    for side in (0,1):
        hk[:,side,0]=j[:,20+side]
        # SMPL-X joint order: index, middle, pinky, ring, thumb.
        for f,digit in enumerate(digits):
            start=25+side*15+3*('index','middle','pinky','ring','thumb').index(digit)
            hk[:,side,1+4*f:4+4*f]=j[:,start:start+3]
            hk[:,side,4+4*f]=vertices[:,vertex_ids['smplx'][('l','r')[side]+digit]]
    packed=np.zeros((T,52,3));body_map={0:0,1:1,4:2,7:3,10:4,2:5,5:6,8:7,11:8,3:9,6:10,9:11,12:12,15:13,13:14,16:15,18:16,20:17,14:33,17:34,19:35,21:36}
    for src,dst in body_map.items():packed[:,dst]=j[:,src]
    for side,offset in ((0,18),(1,37)):
        for k,f in enumerate((1,2,4,3,0)):packed[:,offset+3*k:offset+3*k+3]=hk[:,side,[1+4*f,2+4*f,4+4*f]]
    with open(r['metadata']['trajectory']) as f:rows=[x for x in csv.DictReader(f) if int(x['frame'])%3==0][:T]
    assert len(rows)==T
    xyz=np.array([[float(x[k]) for k in ('px','py','pz')] for x in rows])@A.T*scale
    rot=A@Rotation.from_quat([[float(x[k]) for k in ('qx','qy','qz','qw')] for x in rows]).as_matrix()
    quat=Rotation.from_matrix(rot).as_quat()[:,[3,0,1,2]]
    mesh=trimesh.load(inputs(seq)['object_mesh'],force='mesh')
    # Same deterministic, evenly spread object samples for every solver candidate.
    np.random.seed(0)
    pts,_=trimesh.sample.sample_surface_even(mesh,100)
    pose=np.c_[xyz,quat]
    return (dict(human_joints_scaled=packed,hand_kp_scaled=hk,object_pose_mujoco=pose,
        object_points_scaled=pts*scale,scale=np.array(scale),z_min=np.array(0.),frame_ids=np.arange(T),
        floor_mode=np.array('source'),human_representation=np.array('HiPHI articulated pose+shape SMPL-X')),
        dict(vertices=vertices.astype(np.float32),faces=model.faces.astype(np.int32),fps=30))

def install_model(source,dest,oldobj,obj,pieces,mesh,scale):
    tree=ET.parse(source);root=tree.getroot();comp=root.find('compiler')
    if comp is None:comp=ET.SubElement(root,'compiler')
    for key in ('meshdir','texturedir'):
        p=Path(comp.get(key,'.'))
        if not p.is_absolute():comp.set(key,str((Path(source).parent/p).resolve()))
    body=root.find(f".//body[@name='{oldobj}_link']");assert body is not None
    assets=root.find('asset')
    for x in list(assets):
        if oldobj in x.get('name',''):assets.remove(x)
    for x in list(body):
        if x.tag=='geom':body.remove(x)
    for node in root.iter():
        for k,v in list(node.attrib.items()):
            if k!='file' and oldobj in v:node.set(k,v.replace(oldobj,obj))
    ET.SubElement(assets,'mesh',name=obj+'_visual_mesh',file=str(mesh),scale=f'{scale} {scale} {scale}')
    ET.SubElement(body,'geom',name=obj+'_visual',type='mesh',mesh=obj+'_visual_mesh',contype='0',conaffinity='0',rgba='.60 .43 .26 1',group='1')
    for i,p in enumerate(pieces):
        name=f'{obj}_piece_{i:03d}';ET.SubElement(assets,'mesh',name=name+'_mesh',file=str(p),scale=f'{scale} {scale} {scale}')
        ET.SubElement(body,'geom',name=name,type='mesh',mesh=name+'_mesh',contype='1',conaffinity='1',rgba='.3 .6 .9 0',friction='.9 .5 .5',group='1')
    dest.parent.mkdir(parents=True,exist_ok=True);tree.write(dest)

def main(seq):
    start=time.perf_counter();r=row(seq);info=inputs(seq);obj=r['object']
    # CoACD can have a much larger temporary peak than the motion solver.
    # Serialize decomposition even across independently launched smoke services.
    with (OUT/'decomposition.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);make_geometry(r)
    z,human=humoto_source(r) if r['dataset']=='humoto' else hiphi_source(r)
    info['canonical'].parent.mkdir(parents=True,exist_ok=True);info['human'].parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(info['canonical'],**z);np.savez_compressed(info['human'],**human)
    mesh=trimesh.load(info['object_mesh'],force='mesh');corners=mesh.convex_hull.vertices*float(z['scale'])
    rotations=Rotation.from_quat(z['object_pose_mujoco'][:,[4,5,6,3]]).as_matrix()
    bottom=np.min(np.einsum('tij,vj->tvi',rotations,corners)[:,:,2]+z['object_pose_mujoco'][:,2,None],axis=1)
    write(OUT/'clips'/seq/'scene_audit.json',dict(scale=float(z['scale']),fps=30,
        human_skin_min_z_mm=float(human['vertices'][:,:,2].min()*1000),object_min_z_mm=float(bottom.min()*1000),
        object_floor_below_1mm_frames=int(np.sum(bottom<-.001)),source_floor_preserved=True,
        note='These are input-scene values, not robot violations. No separate human/object regrounding. Fitted HiPHI skin may cross the original floor.'))
    template='drag_utility_cart_on_ground_while_walking_back_and_forth-677'
    nativejob=read(NATIVE/'jobs'/f'{template}.json')
    nativez=np.load(NATIVE/'clips'/template/'part0'/f'{template}_canonical.npz')
    bodybase=Path(str(nativez['stageA_model_base']));fullbase=Path(nativejob['model_xml'])
    pieces=sorted((OUT/'geometry'/obj).glob('piece_*.obj'))
    unscaled=OUT/'clips'/seq/'unscaled'/f'g1_29dof_wuji_welded_w_{obj}.xml'
    install_model(bodybase,unscaled,'utility_cart',obj,pieces,info['object_mesh'],1.)
    cfg=PipelineConfig(task_name=seq,object_name=obj,save_dir=OUT/'clips'/seq,stageA_model_base=unscaled)
    _,scaled,_=build_scaled_model(cfg,'A',float(z['scale']));assert Path(scaled)==info['body_model']
    install_model(fullbase,info['model'],'utility_cart',obj,pieces,info['object_mesh'],float(z['scale']))
    import mujoco
    coverage=validate_body_object_colliders(mujoco.MjModel.from_xml_path(scaled),obj)
    m=mujoco.MjModel.from_xml_path(str(info['model']));assert m.nq==83,m.nq
    write(OUT/'jobs'/f'{seq}.json',dict(sequence=seq,object=obj,model_xml=str(info['model']),
        final_npz=str(destination(seq)/'final_v2.npz'),body_pen_tolerance=.001))
    write(marker(seq,'canonical'),dict(sequence=seq,frames=len(z['hand_kp_scaled']),seconds=time.perf_counter()-start,
        unscaled_body_model=str(unscaled),coverage=coverage,scale=float(z['scale']),source=r,
        model_sha256=sha(info['model']),canonical_sha256=sha(info['canonical'])))

if __name__=='__main__':run_stage('canonical',main,sys.argv[1])
