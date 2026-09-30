"""Render source, world-scaled, and local-scaled OmniRetarget with one camera."""
from contextlib import ExitStack
from pathlib import Path
import hashlib,importlib.util,json,shutil,subprocess,sys
import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco,numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

CODE=Path(__file__).resolve().parent;ROOT=CODE.parents[2]
OUT=ROOT/'retarget_eval/omomo_largebox_scaled_frames';RENDERS=OUT/'renders';RENDERS.mkdir(parents=True,exist_ok=True)
BASE=ROOT/'retarget_eval/omomo_scaled_world_30fps_90';SOURCE_RENDER=BASE/'renders'
SEQ='sub10_largebox_061';OBJECT='largebox'
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
base=module('omomo_existing_renderer',ROOT/'UMR-cmp-omomo-scaled/experiments/omomo_scaled_world_30fps_90/render_results.py')
worldpath=BASE/'omni'/f'{SEQ}_original.npz';localpath=OUT/'object_frame'/f'{SEQ}_original.npz'
with np.load(worldpath) as w,np.load(localpath) as l:
 q_world=w['qpos'].astype(float);q_local=l['qpos'].astype(float)
assert q_world.shape==q_local.shape==(117,43)
input_path=ROOT/'retarget_eval/npz/smplx_hoi'/f'{SEQ}.npz'
with np.load(input_path) as inp:az=base.viz.view_azimuth(inp['global_joint_positions']);scale=1.32/float(inp['height'])
source=SOURCE_RENDER/f'{SEQ}_source_umr_omni.mp4';world_solo=SOURCE_RENDER/f'{SEQ}_omni.mp4'
assert source.exists() and world_solo.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
oldpkg=ROOT/'holosoma-cmp-omomo-scaled/src/holosoma_retargeting/holosoma_retargeting'
newpkg=ROOT/'holosoma-cmp-largebox-frames/src/holosoma_retargeting/holosoma_retargeting'
for rel in [f'models/g1/g1_29dof_w_{OBJECT}.xml',f'models/{OBJECT}/{OBJECT}.obj']:
 assert sha(oldpkg/rel)==sha(newpkg/rel),rel
model=base.omni_model(OBJECT,scale)
model.vis.quality.shadowsize=1024

def floor_clearances(q):
 scene=Path(base.scale.scaled_scene(oldpkg/f'models/g1/g1_29dof_w_{OBJECT}.xml',OBJECT,scale,SOURCE_RENDER/'scenes'))
 model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
 ground=model.geom('ground').id
 foot={side:[] for side in ('left','right')}
 for gid in range(model.ngeom):
  body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[gid])) or ''
  for side in foot:
   if body.startswith(side+'_ankle_roll') and model.geom_contype[gid]:foot[side].append(gid)
 assert all(foot.values())
 objbody=model.body('largebox_link').id
 object_geoms=[gid for gid in range(model.ngeom) if model.geom_bodyid[gid]==objbody]
 assert len(object_geoms)==1
 obj_geom=object_geoms[0];mid=int(model.geom_dataid[obj_geom]);vert=model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
 rows=[];fromto=np.zeros(6)
 for frame in q:
  data.qpos[:]=frame;mujoco.mj_forward(model,data)
  feet={side:min(float(mujoco.mj_geomDistance(model,data,ground,g,1.,fromto)) for g in geoms)*1000 for side,geoms in foot.items()}
  lowest=float((vert@data.geom_xmat[obj_geom].reshape(3,3).T+data.geom_xpos[obj_geom])[...,2].min()*1000)
  rows.append(dict(object_bottom_mm=lowest,left_foot_mm=feet['left'],right_foot_mm=feet['right']))
 return rows
world_heights=floor_clearances(q_world);local_heights=floor_clearances(q_local)
measurements=dict(sequence=SEQ,fps=30,frames=len(q_world),object_scale=scale,world=world_heights,object_frame=local_heights,
                   definition='signed nearest geometry-to-floor distance in mm for collision foot geoms; object is lowest visual-mesh vertex in world coordinates; positive=above floor, negative=penetration',
                   object_z_difference_mm=float(np.mean((q_local[:,38]-q_world[:,38])*1000)),
                   object_z_difference_range_mm=[float(np.min((q_local[:,38]-q_world[:,38])*1000)),float(np.max((q_local[:,38]-q_world[:,38])*1000))])
(RENDERS/'floor_clearance.json').write_text(json.dumps(measurements,indent=2)+'\n')
T=len(q_world)
def frame_local(m,d,t):d.qpos[:]=q_local[t]
world_copy=RENDERS/f'{SEQ}_world_frame.mp4';shutil.copy2(world_solo,world_copy)
combined=RENDERS/f'{SEQ}_source_world_object.mp4';single=RENDERS/f'{SEQ}_object_frame.mp4'
indices=[0,T//4,T//2,3*T//4,T-1];stills={}
with ExitStack() as stack:
 panel=base.Panel(model,q_world[:,:3],az,scale,frame_local);stack.callback(panel.close)
 reader=stack.enter_context(imageio.get_reader(source,input_params=['-threads','1']))
 assert abs(reader.get_meta_data()['fps']-30)<1e-6
 wc=stack.enter_context(imageio.get_writer(str(combined)+'.partial.mp4',fps=30,codec='libx264',pixelformat='yuv420p',quality=8,macro_block_size=8,ffmpeg_params=['-threads','1','-preset','veryfast']))
 wo=stack.enter_context(imageio.get_writer(str(single)+'.partial.mp4',fps=30,codec='libx264',pixelformat='yuv420p',quality=8,macro_block_size=8,ffmpeg_params=['-threads','1','-preset','veryfast']))
 for t in range(T):
  old=reader.get_next_data();assert old.shape==(base.viz.H,base.viz.W*3,3),old.shape
  source_panel=old[:,:base.viz.W].copy();world_panel=old[:,base.viz.W*2:].copy()
  obj=local_heights[t]['object_bottom_mm'];feet=min(local_heights[t]['left_foot_mm'],local_heights[t]['right_foot_mm'])
  local_panel=base.viz.label(panel.render(t),[("OmniRetarget: scaled mesh + native Z",17),
     (f"Object bottom {obj:+.0f} mm; lower foot {feet:+.0f} mm",15),
     (f"Frame {t+1}/{T} | {t/30:.2f} s",15)])
  frame=np.hstack([source_panel,world_panel,local_panel]);wc.append_data(frame);wo.append_data(local_panel)
  if t in indices:stills[t]=frame.copy()
  if t==0:Image.fromarray(frame).save(RENDERS/f'{SEQ}_first_frame.png')
  if t==T//2:Image.fromarray(frame).save(RENDERS/f'{SEQ}_mid_frame.png')
  if t%30==0:print('Rendered',t+1,'/',T,flush=True)
 backend=panel.backend
Path(str(combined)+'.partial.mp4').replace(combined);Path(str(single)+'.partial.mp4').replace(single)
sheet=Image.new('RGB',(1920,272*len(indices)),(235,237,240))
for i,t in enumerate(indices):sheet.paste(Image.fromarray(stills[t]).resize((1920,272)),(0,i*272))
sheet.save(RENDERS/f'{SEQ}_contact_sheet.png')
# Verify complete frames, FPS and dimensions for all three standalone artifacts.
import re
for path,size in [(combined,(1920,544)),(single,(640,544)),(world_copy,(640,544))]:
 reader=imageio_ffmpeg.read_frames(str(path),input_params=['-threads','1']);meta=next(reader);reader.close()
 assert abs(meta['fps']-30)<1e-6 and tuple(meta['size'])==size,(path,meta)
 done=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-threads','1','-i',str(path),'-map','0:v:0','-f','null','-','-progress','pipe:1','-nostats'],capture_output=True,text=True,check=True)
 counts=re.findall(r'^frame=(\d+)',done.stdout,re.M);assert counts and int(counts[-1])==T,(path,counts)
assert Image.open(RENDERS/f'{SEQ}_first_frame.png').size==(1920,544)
assert Image.open(RENDERS/f'{SEQ}_mid_frame.png').size==(1920,544)
summary={}
for mode,values in [('world',world_heights),('object_frame',local_heights)]:
 summary[mode]=dict(first_frame_object_bottom_mm=values[0]['object_bottom_mm'],first_frame_left_foot_mm=values[0]['left_foot_mm'],first_frame_right_foot_mm=values[0]['right_foot_mm'],median_object_bottom_mm=float(np.median([v['object_bottom_mm'] for v in values])),median_min_foot_mm=float(np.median([min(v['left_foot_mm'],v['right_foot_mm']) for v in values])))
summary.update(object_frame_minus_world_z_mm=measurements['object_z_difference_mm'],fps=30,frames=T,renderer=backend,world_output_sha256=sha(worldpath),object_output_sha256=sha(localpath),combined_mp4=combined.name,world_mp4=world_copy.name,object_mp4=single.name)
(RENDERS/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OMOMO largebox scaling frames</title><style>body{font:16px system-ui;background:#10161f;color:#ebf2fb;max-width:1500px;margin:auto;padding:28px}p{line-height:1.55;color:#b7c8d8}a{color:#8fc7ff;margin-right:20px}video,img{width:100%;border-radius:8px}main{background:#1a2533;padding:18px;border-radius:12px}table{border-collapse:collapse;margin:20px 0}td,th{border:1px solid #536a82;padding:10px 15px}nav{margin:22px 0}</style><h1>OMOMO largebox: two ways to scale the object</h1><p>Sequence sub10_largebox_061, 117 frames at 30 FPS. Left: unchanged SMPL-X source. Middle: OmniRetarget with mesh and object trajectory scaled together in the world frame. Right: the same scaled mesh with OmniRetarget’s original first-frame object-height rule. Both retargeted outputs were solved separately with identical human input, object mesh scale and solver settings.</p><main><video controls playsinline preload="metadata" poster="sub10_largebox_061_mid_frame.png" src="sub10_largebox_061_source_world_object.mp4"></video><nav><a href="sub10_largebox_061_source_world_object.mp4" download>Combined MP4</a><a href="sub10_largebox_061_world_frame.mp4" download>World-frame MP4</a><a href="sub10_largebox_061_object_frame.mp4" download>Object-frame MP4</a><a href="sub10_largebox_061_first_frame.png">First-frame PNG</a><a href="sub10_largebox_061_mid_frame.png">Mid-frame PNG</a><a href="sub10_largebox_061_contact_sheet.png">Contact sheet</a></nav></main><p>Clearance is the signed distance from the object’s lowest mesh vertex or robot foot collision surface to the floor. Positive means above floor. See <a href="summary.json">measurements</a> and <a href="../PROTOCOL.md">protocol</a> for the exact scene and scaling rules.</p><table><tr><th>Mode</th><th>First object clearance</th><th>First left foot</th><th>First right foot</th><th>Median lower-foot clearance</th></tr>__ROWS__</table><img src="sub10_largebox_061_contact_sheet.png" alt="Five frames through the sequence"></html>'''
rows=''.join(f'<tr><td>{mode}</td><td>{v["first_frame_object_bottom_mm"]:+.1f} mm</td><td>{v["first_frame_left_foot_mm"]:+.1f} mm</td><td>{v["first_frame_right_foot_mm"]:+.1f} mm</td><td>{v["median_min_foot_mm"]:+.1f} mm</td></tr>' for mode,v in summary.items() if mode in ('world','object_frame'))
(RENDERS/'index.html').write_text(html.replace('__ROWS__',rows))
print('Rendered and validated:',RENDERS/'index.html',summary,flush=True)
