"""One CPU worker; persistent per-attempt records and resumable output validation."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path
import numpy as np
CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[2]
OUT=ROOT/'retarget_eval/omomo_scaled_object_frame_30fps_90'
UMR=ROOT/'UMR-cmp-omomo-scaled'
OMNI=ROOT/'holosoma-cmp-omomo-object-frame/src/holosoma_retargeting/holosoma_retargeting'
PY_U=Path('/home/natcha/miniconda3/envs/umr/bin/python')
PY_O=Path('/home/natcha/miniconda3/envs/omniretarget/bin/python')
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',RAYON_NUM_THREADS='1',MUJOCO_GL='disable',PYTHONUNBUFFERED='1')
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def validate(path,frames,width):
 with np.load(path,allow_pickle=True) as d:
  assert d['qpos'].shape==(frames,width),(path,d['qpos'].shape,frames)
  assert np.isfinite(d['qpos']).all()
  assert float(np.asarray(d['fps']).reshape(-1)[0])==30
  if width==36:
   assert np.array_equal(d['frame_ids'],np.arange(frames))
   assert bool(np.asarray(d['zero_source_finger_pose']).reshape(-1)[0])
 return sha(path)
def available_gb():
 for line in Path('/proc/meminfo').read_text().splitlines():
  if line.startswith('MemAvailable:'):return int(line.split()[1])/1024**2

def run(seq,obj,method):
 frames=len(np.load(ROOT/'UMR/sample_data/omomo/train_and_test'/seq/'poses.npy',mmap_mode='r'))
 dest=OUT/method/(f'{seq}.npz' if method=='umr' else f'{seq}_original.npz')
 record=OUT/'attempts'/f'{seq}_{method}.json'
 if record.exists():
  old=json.loads(record.read_text())
  if old['status']=='success':assert validate(dest,frames,36 if method=='umr' else 43)==old['sha256']
  if old['status'] in ('success','solver_failed'):return
 while available_gb()<8:
  write(OUT/'status.json',dict(status='waiting_for_memory',sequence=seq,method=method,available_gib=available_gb()))
  time.sleep(30)
 dest.parent.mkdir(parents=True,exist_ok=True)
 if method=='umr':
  cmd=[str(PY_U),'scripts/humanoid_retarget_pipeline_hsi_hoi.py','--config','robot_configs/humanoid_retarget_unitree_g1_example.json','--defaults',str(CODE/'umr_cpu_defaults.json'),'--data',str(ROOT/'UMR/sample_data/omomo'),'--seq-key',seq,'--stage','retarget','--skip-view','--force-retarget','--out',str(dest)]
  cwd=UMR
 else:
  cmd=[str(PY_O),str(CODE/'run_omni.py'),'--task-name',seq,'--task-type','object_interaction','--robot','g1','--data-format','smplx_hand','--task-config.object-name',obj,'--task-config.object-scale-mode','object_scaled','--data-path',str(ROOT/'retarget_eval/npz/smplx_hoi'),'--save-dir',str(dest.parent)]
  cwd=OMNI
 log=OUT/'logs'/f'{seq}_{method}.log';log.parent.mkdir(exist_ok=True)
 start=time.time();reason=None
 write(OUT/'status.json',dict(status='running',sequence=seq,method=method,started=time.strftime('%Y-%m-%d %H:%M:%S'),pid=os.getpid()))
 with log.open('w') as f:
  p=subprocess.Popen(cmd,cwd=cwd,env=ENV,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  while p.poll() is None:
   if available_gb()<4.5:reason='infrastructure: stopped comparison child to preserve system memory'
   if time.time()-start>3600:reason='infrastructure: timeout after 3600 seconds'
   if reason:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    break
   with (OUT/'resource_log.jsonl').open('a') as monitor:
    cg=Path('/proc/self/cgroup').read_text().strip().split('::')[-1]
    cp=Path('/sys/fs/cgroup')/cg.lstrip('/')
    monitor.write(json.dumps(dict(time=time.time(),sequence=seq,method=method,available_gib=available_gb(),comparison_memory_bytes=int((cp/'memory.current').read_text()) if (cp/'memory.current').exists() else None,training_pid_present=Path('/proc/1151374').exists()))+'\n')
   time.sleep(2)
  rc=p.wait()
 status='success' if rc==0 and dest.exists() else 'solver_failed'
 digest=None
 if status=='success':
  digest=validate(dest,frames,36 if method=='umr' else 43)
  if method=='umr':assert 'retarget_object_size=scaled' in log.read_text()
 else:
  lines=log.read_text(errors='replace').splitlines()
  reason=reason or next((s for s in reversed(lines) if 'Error:' in s or 'Exception:' in s),'See log')
  if not ('CVXPY solve failed' in reason or 'infeasible' in reason.lower()):status='infrastructure_failed'
 write(record,dict(sequence=seq,object=obj,method=method,status=status,returncode=rc,seconds=time.time()-start,frames=frames,fps=30,command=cmd,cwd=str(cwd),output=str(dest),sha256=digest,reason=reason,log=str(log)))
 print(seq,method,status,round(time.time()-start,1),reason or '',flush=True)
 if reason and reason.startswith('infrastructure:'):status='infrastructure_failed'
 if status=='infrastructure_failed':
  write(OUT/'status.json',dict(status='infrastructure_failed',sequence=seq,method=method,reason=reason))
  raise RuntimeError(reason)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=90);args=parser.parse_args()
 manifest=json.loads((CODE/'sequence_manifest.json').read_text())
 sequences=[(s,o) for o,ss in manifest['objects'].items() for s in ss][:args.limit]
 for seq,obj in sequences:
  run(seq,obj,'omni')
  evaluator=CODE/'evaluate.py'
  if evaluator.exists():subprocess.run([str(PY_O),str(evaluator),'--sequence',seq],env=ENV,check=True)
 write(OUT/'status.json',dict(status='complete' if args.limit==90 else 'smoke_complete',sequences=len(sequences),finished=time.strftime('%Y-%m-%d %H:%M:%S')))
