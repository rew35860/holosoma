"""Freeze the old UMR arm and restore the six native object scenes locally."""
from pathlib import Path
import hashlib,json,shutil,subprocess
import numpy as np
CODE=Path(__file__).resolve().parent;ROOT=CODE.parents[2]
OUT=ROOT/'retarget_eval/omomo_scaled_object_frame_30fps_90';BASE=ROOT/'retarget_eval/omomo_scaled_world_30fps_90'
OLD=ROOT/'holosoma-cmp-omomo-scaled/src/holosoma_retargeting/holosoma_retargeting';NEW=ROOT/'holosoma-cmp-omomo-object-frame/src/holosoma_retargeting/holosoma_retargeting'
manifest=json.loads((CODE/'sequence_manifest.json').read_text());prior=json.loads((ROOT/'UMR-cmp-omomo-scaled/experiments/omomo_scaled_world_30fps_90/input_audit.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
OUT.mkdir(exist_ok=True)
assets=[]
for obj in manifest['objects']:
 for rel in [f'models/{obj}/{obj}.obj',f'models/g1/g1_29dof_w_{obj}.xml']:
  src=OLD/rel;dst=NEW/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  if not dst.exists():shutil.copy2(src,dst)
  assert sha(src)==sha(dst),(src,dst)
  assets.append(dict(source=str(src),destination=str(dst),sha256=sha(src)))
(CODE/'local_assets.json').write_text(json.dumps(assets,indent=2)+'\n')
(OUT/'local_assets.json').write_text(json.dumps(assets,indent=2)+'\n')
rows=[]
for obj,seqs in manifest['objects'].items():
 assert len(seqs)==15
 for seq in seqs:
  inp=ROOT/'retarget_eval/npz/smplx_hoi'/f'{seq}.npz';oldrec=BASE/'attempts'/f'{seq}_umr.json';rec=json.loads(oldrec.read_text())
  assert rec['status']=='success' and rec['method']=='umr'
  oldout=Path(rec['output']);assert sha(oldout)==rec['sha256']
  with np.load(oldout,allow_pickle=True) as d:
   T=len(d['qpos']);assert d['qpos'].shape==(T,36) and float(d['fps'][0])==30
   assert np.array_equal(d['frame_ids'],np.arange(T)) and bool(d['zero_source_finger_pose'][0])
  baseaudit=next(r for r in prior if r['sequence']==seq)
  assert baseaudit['input_sha256']==sha(inp) and baseaudit['frames']==T
  dst=OUT/'attempts'/oldrec.name;dst.parent.mkdir(exist_ok=True)
  if not dst.exists():shutil.copy2(oldrec,dst)
  assert sha(dst)==sha(oldrec)
  rows.append(dict(sequence=seq,object=obj,frames=T,input_sha256=sha(inp),umr_output_sha256=rec['sha256'],umr_attempt_sha256=sha(oldrec),source_cache_sha256=baseaudit['source_cache_sha256']))
assert len(rows)==90 and sum(r['frames'] for r in rows)==16804
(OUT/'input_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
provenance=dict(sequences=90,frames=16804,fps=30,umr_outputs_reused=90,baseline_umr_validation_sha256=sha(BASE/'final_validation.json'),frozen_metric_sha256=sha(ROOT/'retarget_eval/reproduction_fullsize_30fps_90/snapshots/eval_main.py'),main_results_sha256=sha(ROOT/'retarget_eval/main_results.json'),branch=subprocess.check_output(['git','-C',str(ROOT/'holosoma-cmp-omomo-object-frame'),'branch','--show-current'],text=True).strip())
(OUT/'preflight.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Verified 90 UMR outputs, 90 identical inputs and six identical native Omni object scenes.',flush=True)
