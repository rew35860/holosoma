"""Retarget the object-frame variant, verify the paired trajectories, then render."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time
import numpy as np
CODE=Path(__file__).resolve().parent;ROOT=CODE.parents[2]
OUT=ROOT/'retarget_eval/omomo_largebox_scaled_frames';OUT.mkdir(parents=True,exist_ok=True)
BASE=ROOT/'retarget_eval/omomo_scaled_world_30fps_90';PKG=ROOT/'holosoma-cmp-largebox-frames/src/holosoma_retargeting/holosoma_retargeting'
SEQUENCE='sub10_largebox_061';OBJECT='largebox'
WORLD=BASE/'omni'/f'{SEQUENCE}_original.npz';LOCAL=OUT/'object_frame'/f'{SEQUENCE}_original.npz'
PY='/home/natcha/miniconda3/envs/omniretarget/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,val):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(val,indent=2)+'\n')
def available():
 for line in Path('/proc/meminfo').read_text().splitlines():
  if line.startswith('MemAvailable:'):return int(line.split()[1])/1024**2
baseline=json.loads((BASE/'attempts'/f'{SEQUENCE}_omni.json').read_text());assert baseline['status']=='success' and sha(WORLD)==baseline['sha256']
source=ROOT/'retarget_eval/npz/smplx_hoi'/f'{SEQUENCE}.npz'
with np.load(source) as a:
 T=len(a['global_joint_positions']);scale=1.32/float(a['height']);obj=a['object_poses'].copy();rawhuman=a['global_joint_positions'].copy()
assert T==117
# The same first-frame object pose is fed to both modes. They use exactly the same mesh scale.
zmin=float(rawhuman[:,[10,11],2].min());zmin-=.1 if zmin>=.1 else 0
expected_world=(obj[:,6]-zmin)*scale
expected_local=obj[0,6]+(obj[:,6]-obj[0,6])*scale
expected_shift=expected_local-expected_world
assert np.max(np.abs(expected_shift-expected_shift[0]))<2e-6
write(OUT/'input_audit.json',dict(sequence=SEQUENCE,frames=T,fps=30,input_sha256=sha(source),world_output_sha256=sha(WORLD),mesh_scale=scale,preprocess_floor_origin_z_m=zmin,object_initial_z_m=float(obj[0,6]),expected_object_frame_minus_world_z_m=float(expected_shift[0]),reference_code_commit='dfd1c2f'))
if not LOCAL.exists():
 while available()<8:write(OUT/'status.json',dict(status='waiting_for_memory',available_gib=available()));time.sleep(20)
 LOCAL.parent.mkdir(parents=True,exist_ok=True)
 cmd=[PY,str(CODE/'run_omni.py'),'--task-name',SEQUENCE,'--task-type','object_interaction','--robot','g1','--data-format','smplx_hand','--task-config.object-name',OBJECT,'--task-config.object-scale-mode','object_scaled','--data-path',str(source.parent),'--save-dir',str(LOCAL.parent)]
 write(OUT/'status.json',dict(status='retargeting',sequence=SEQUENCE,frames=T))
 with (OUT/'retarget.log').open('w') as log:
  child=subprocess.Popen(cmd,cwd=PKG,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',MUJOCO_GL='disable',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',RAYON_NUM_THREADS='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  start=time.time()
  while child.poll() is None:
   if available()<4.5 or time.time()-start>1800:
    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
    write(OUT/'status.json',dict(status='stopped_to_protect_training' if available()<4.5 else 'timeout',seconds=time.time()-start));raise RuntimeError('Comparison child stopped; see status.json')
   time.sleep(3)
  if child.returncode:write(OUT/'status.json',dict(status='retarget_failed',returncode=child.returncode));raise RuntimeError('OmniRetarget failed; see retarget.log')
with np.load(LOCAL) as l,np.load(WORLD) as w:
 q=l['qpos'];qw=w['qpos'];assert q.shape==qw.shape==(T,43)
 assert np.isfinite(q).all() and np.isfinite(qw).all()
 np.testing.assert_allclose(l['human_joints'],w['human_joints'],atol=2e-6)
 np.testing.assert_allclose(q[:,36:38],qw[:,36:38],atol=2e-6)
 np.testing.assert_allclose(q[:,38],expected_local,atol=2e-6)
 np.testing.assert_allclose(qw[:,38],expected_world,atol=2e-6)
 np.testing.assert_allclose(q[:,39:43],qw[:,39:43],atol=2e-6)
 # Both modes retain the same collision mesh, so difference must be the target Z rule.
 assert np.max(np.abs((q[:,38]-qw[:,38])-expected_shift))<2e-6
write(OUT/'trajectory_validation.json',dict(object_frame_output_sha256=sha(LOCAL),world_output_sha256=sha(WORLD),frames=T,same_human_targets=True,same_object_xy=True,same_object_rotation=True,same_scaled_mesh=True,object_frame_minus_world_z_mm=float(expected_shift[0]*1000)))
write(OUT/'status.json',dict(status='rendering',sequence=SEQUENCE,frames=T))
render=CODE/'render_compare.py'
subprocess.run([PY,str(render)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',EGL_PLATFORM='surfaceless',LIBGL_ALWAYS_SOFTWARE='1',MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',GALLIUM_DRIVER='llvmpipe',LP_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_LIBRARY_PATH='/home/natcha/miniconda3/envs/osmesa/lib',__EGL_VENDOR_LIBRARY_FILENAMES='/home/natcha/miniconda3/envs/osmesa/share/glvnd/egl_vendor.d/50_mesa.json'),cwd=PKG,check=True)
write(OUT/'status.json',dict(status='complete',sequence=SEQUENCE,frames=T,finished=time.strftime('%Y-%m-%d %H:%M:%S')))
