"""CPU-only synchronized original human / existing v1 / v2 mesh comparison."""
from pathlib import Path
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
os.environ.update(CUDA_VISIBLE_DEVICES='',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
    __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json',
    EGL_PLATFORM='surfaceless',LIBGL_ALWAYS_SOFTWARE='1',GALLIUM_DRIVER='llvmpipe',LP_NUM_THREADS='1')
os.environ.pop('IMAGEIO_FFMPEG_EXE',None)
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from catalog import OUT, baseline_for

def render(seq,preview=False):
    start=time.perf_counter();base=baseline_for(seq)
    job=json.loads((base/'jobs'/f'{seq}.json').read_text())
    raw=np.load(OUT/'human'/f'{seq}.npz');vertices=raw['vertices'];faces=raw['faces']
    q=[np.load(job['final_npz'])['qpos'],np.load(OUT/'clips'/seq/'final_v2.npz')['qpos']]
    assert len(vertices)==len(q[0])==len(q[1])
    folder=OUT/'renders_human';folder.mkdir(exist_ok=True)
    assets=folder/'assets'/seq;assets.mkdir(parents=True,exist_ok=True)
    objfile=assets/'human_frame0.obj'
    if not objfile.exists():
        with objfile.open('w') as f:
            np.savetxt(f,vertices[0],fmt='v %.9g %.9g %.9g')
            np.savetxt(f,faces+1,fmt='f %d %d %d')
    tree=ET.parse(job['model_xml']);root=tree.getroot()
    compiler=root.find('compiler')
    if compiler is not None:
        for key in ('meshdir','texturedir'):
            value=compiler.get(key)
            if value and not Path(value).is_absolute():compiler.set(key,str((Path(job['model_xml']).parent/value).resolve()))
    ET.SubElement(root.find('asset'),'mesh',name='source_human_mesh',file=str(objfile))
    ET.SubElement(root.find('worldbody'),'geom',name='source_human_geom',type='mesh',mesh='source_human_mesh',
                  rgba='.40 .68 .88 1',contype='0',conaffinity='0',group='3')
    xml=assets/'three_panel.xml';tree.write(xml)
    m=mujoco.MjModel.from_xml_path(str(xml));m.vis.global_.offwidth=480;m.vis.global_.offheight=480
    d=mujoco.MjData(m);renderer=mujoco.Renderer(m,480,480)
    mid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_MESH,'source_human_mesh')
    gid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'source_human_geom')
    va,vn=int(m.mesh_vertadr[mid]),int(m.mesh_vertnum[mid])
    fa,fn=int(m.mesh_faceadr[mid]),int(m.mesh_facenum[mid])
    na,nn=int(m.mesh_normaladr[mid]),int(m.mesh_normalnum[mid])
    rotation=Rotation.from_quat(m.mesh_quat[mid][[1,2,3,0]]).as_matrix()
    original=m.mesh_vert[va:va+vn]@rotation.T+m.mesh_pos[mid]
    error,remap=cKDTree(vertices[0]).query(original)
    assert error.max()<2e-5,error.max()
    meshfaces=m.mesh_face[fa:fa+fn].copy();normalids=m.mesh_facenormal[fa:fa+fn].copy()
    for g in range(m.ngeom):
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
        body=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or ''
        m.geom_group[g]=3 if g==gid else 2 if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE else 1 if body.startswith(job['object']) else 0
    visible=np.flatnonzero((m.geom_type!=mujoco.mjtGeom.mjGEOM_PLANE)&(m.geom_rgba[:,3]>0)&(np.arange(m.ngeom)!=gid))
    centers=[];radius=0.
    for t in range(len(vertices)):
        center=.5*(q[0][t,:3]+q[0][t,-7:-4]);center[2]=.8;centers.append(center)
        radius=max(radius,float(np.linalg.norm(vertices[t]-center,axis=1).max()))
        for traj in q:
            d.qpos[:]=traj[t];mujoco.mj_kinematics(m,d)
            radius=max(radius,float(np.max(np.linalg.norm(d.geom_xpos[visible]-center,axis=1)+m.geom_rbound[visible])))
    cam=mujoco.MjvCamera();cam.azimuth=210;cam.elevation=-20
    cam.distance=max(2.5,radius*1.06/np.sin(np.deg2rad(m.vis.global_.fovy/2)))
    normal_distance=cam.distance
    option=mujoco.MjvOption()
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    hk=np.load(base/'clips'/seq/'part0'/f'{seq}_canonical.npz')['hand_kp_scaled']
    def upload(t):
        v=(vertices[t,remap]-m.mesh_pos[mid])@rotation
        m.mesh_vert[va:va+vn]=v
        triangles=v[meshfaces]
        normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        accum=np.zeros((nn,3))
        for k in range(3):np.add.at(accum,normalids[:,k],normals)
        accum/=np.maximum(np.linalg.norm(accum,axis=1,keepdims=True),1e-12)
        m.mesh_normal[na:na+nn]=accum
        mujoco.mjr_uploadMesh(m,renderer._mjr_context,mid)
    def frame(t,close=False):
        upload(t);canvas=Image.new('RGB',(1440,540),(245,247,250));draw=ImageDraw.Draw(canvas)
        cam.lookat[:]=hk[t,:, [4,8,12,16,20],:].reshape(-1,3).mean(0) if close else centers[t]
        cam.distance=1.1 if close else normal_distance
        for i in range(3):
            d.qpos[:]=q[max(i-1,0)][t];mujoco.mj_forward(m,d)
            option.geomgroup[:]=1;option.geomgroup[0]=int(i!=0);option.geomgroup[3]=int(i==0)
            renderer.update_scene(d,cam,scene_option=option)
            # Equal visualization for all panels; costly mirror/shadow passes are unnecessary for comparison.
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            canvas.paste(Image.fromarray(renderer.render()),(480*i,40))
            draw.text((480*i+10,10),('Human demonstration (scaled)','V1: existing reference','V2: joint arms + fingers')[i],fill=(20,30,45),font=font)
        draw.text((12,519),f'{seq}  |  {t/30:.2f}s / {(len(vertices)-1)/30:.2f}s',fill=(20,30,45),font=font)
        return np.asarray(canvas)
    cache=np.load(base/'evaluation_v1/per_frame'/f'{seq}.npz')
    active=cache['source_contact'].sum(axis=(1,2));ids=np.flatnonzero(active==active.max());t=int(ids[len(ids)//2])
    imageio.imwrite(folder/f'{seq}.png',frame(t))
    imageio.imwrite(folder/f'{seq}_hands.png',frame(t,True))
    if not preview:
        with imageio.get_writer(folder/f'{seq}.mp4',fps=15,codec='libx264',quality=8,macro_block_size=1,
                               ffmpeg_params=['-threads','1','-movflags','+faststart']) as writer:
            for t in range(0,len(vertices),2):writer.append_data(frame(t))
    renderer.close()
    result=dict(sequence=seq,frames_source=len(vertices),video_frames=len(range(0,len(vertices),2)),video_fps=15,
        source_fps=30,panels=['original_human_scaled','existing_v1','v2'],preview_only=preview,
        camera='same camera/object pose/time across panels',human_mesh_initial_mapping_error_m=float(error.max()),
        render_seconds=time.perf_counter()-start)
    (folder/f'{seq}{"_preview" if preview else ""}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':render(sys.argv[1],preview='--preview' in sys.argv)
