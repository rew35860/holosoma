"""Frozen metric formulas, with scene geometry scaled to each method's target."""
import argparse,ast,csv,hashlib,importlib.util,json,math,sys,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
CODE=Path(__file__).resolve().parent;ROOT=CODE.parents[2]
OUT=ROOT/'retarget_eval/omomo_scaled_object_frame_30fps_90'
OLD=ROOT/'retarget_eval/reproduction_fullsize_30fps_90'
CMP=ROOT/'holosoma_cmp/src/holosoma_retargeting/holosoma_retargeting'
PKG=ROOT/'holosoma-cmp-omomo-object-frame/src/holosoma_retargeting/holosoma_retargeting'
assert mujoco.__version__=='3.10.0',mujoco.__version__
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
legacy=module('frozen_metrics',OLD/'snapshots/eval_main.py')
legacy.OMOMO=OLD/'raw_meshes'
scaling=module('world_scaling_eval',PKG/'src/world_scaling.py')
geometry={x['object']:x for x in json.loads((OLD/'geometry_audit.json').read_text())}
values={}
for node in ast.parse((CMP/'config_types/data_type.py').read_text()).body:
 if isinstance(node,ast.Assign):
  for t in node.targets:
   if isinstance(t,ast.Name) and t.id in ('SMPLX_DEMO_JOINTS','JOINTS_MAPPINGS'):values[t.id]=ast.literal_eval(node.value)
BODY=values['SMPLX_DEMO_JOINTS'];MAPPING=values['JOINTS_MAPPINGS'][('smplx_hand','g1')]
NAMES=['Pelvis']+[n for n in MAPPING if n!='Pelvis']
MANIFEST=json.loads((CODE/'sequence_manifest.json').read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,(float,np.floating)):return float(x) if math.isfinite(x) else None
 if isinstance(x,np.integer):return int(x)
 return x

def write(path,data):
 path.parent.mkdir(exist_ok=True,parents=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(clean(data),indent=2,allow_nan=False)+'\n');tmp.replace(path)
def source(seq,obj):
 d=ROOT/'UMR/sample_data/omomo/train_and_test'/seq
 files=[d/x for x in ('poses.npy','transl.npy','betas.npy','gender.npy')]+[d/f'prop_{obj}.csv',d/f'{obj}.xml',OLD/'raw_meshes'/f'{obj}_cleaned_simplified.obj',OLD/'snapshots/eval_main.py']
 fingerprint=hashlib.sha256(('seed=0;'+''.join(sha(p) for p in files)).encode()).hexdigest()
 path=OLD/'source_cache'/f'{seq}.npz'
 with np.load(path) as data:
  assert str(data['fingerprint'])==fingerprint,('source cache fingerprint changed',seq)
  src={k:data[k] for k in ('gap','contact','joints','stance','T')}
 return src,sha(path)
def scene(obj,factor):
 path=scaling.scaled_scene(CMP/f'models/g1/g1_29dof_w_{obj}.xml',obj,factor,OUT/'evaluation_scenes')
 model=mujoco.MjModel.from_xml_path(path)
 name=lambda g:mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
 body=lambda g:mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[g]) or ''
 og={g for g in range(model.ngeom) if name(g).startswith(obj+'_coacd_')}
 fg={g for g in range(model.ngeom) if int(model.geom_type[g])==0}
 rg={g for g in range(model.ngeom) if body(g) not in ('world',obj+'_link') and (model.geom_contype[g] or model.geom_conaffinity[g])}
 hg={g for g in rg if any(h in name(g) for h in legacy.HAND_GEOM)}
 assert og and len(hg)==2 and not (rg&og) and not (rg&fg)
 return model,og,rg,hg,fg

def joint_error(q,model,human):
 data=mujoco.MjData(model);ids=[model.body(MAPPING[n]).id for n in NAMES];idx=[BODY.index(n) for n in NAMES]
 target=human[:,idx[1:]]-human[:,idx[0],None,:];errors=[]
 for t,frame in enumerate(q):
  data.qpos[:]=frame;mujoco.mj_kinematics(model,data)
  errors.extend(np.linalg.norm(data.xpos[ids[1:]]-data.xpos[ids[0]]-target[t],axis=1)*1000)
 return dict(joint_error_mm=float(np.mean(errors)),joint_error_p95_mm=float(np.percentile(errors,95)),joint_landmark_count=14)

def run(seq):
 obj=seq.split('_')[1];src,cache_sha=source(seq,obj);frames=int(src['T'])
 row=dict(sequence=seq,object=obj,frames=frames,fps=30,evaluation_mujoco=mujoco.__version__,source_cache_sha256=cache_sha,split=MANIFEST['splits'][seq])
 d=ROOT/'UMR/sample_data/omomo/train_and_test'/seq
 for method in ('umr','omni'):
  attempt=json.loads((OUT/'attempts'/f'{seq}_{method}.json').read_text());row[method+'_ok']=attempt['status']=='success';row[method+'_attempt']=attempt
  if not row[method+'_ok']:continue
  path=Path(attempt['output']);assert sha(path)==attempt['sha256']
  with np.load(path,allow_pickle=True) as data:
   q=data['qpos'].astype(float)
   if method=='umr':
    s=float(np.asarray(data['smpl_scale']).reshape(-1)[0])
    assert 'retarget_object_size=scaled' in Path(attempt['log']).read_text()
    rows=list(csv.DictReader((d/f'prop_{obj}.csv').open()))
    pos=np.array([[float(r[k]) for k in ('px','py','pz')] for r in rows])*s
    quat=np.array([[float(r[k]) for k in ('qx','qy','qz','qw')] for r in rows])
    object_scale=float(ET.parse(d/f'{obj}.xml').find(f".//mesh[@name='{obj}_visual_mesh']").get('scale').split()[0])
    factor=s*object_scale/geometry[obj]['baked_scale']
    pos-=Rotation.from_quat(quat).apply(np.asarray(geometry[obj]['origin_offset'])*factor)
    q=np.hstack([q,pos,np.roll(quat,1,axis=1)]);human=src['joints']*s
    extra=dict(object_error_translation_mm=0.,object_error_kind='prescribed trajectory')
   else:
    inp=np.load(ROOT/'retarget_eval/npz/smplx_hoi'/f'{seq}.npz');s=1.32/float(inp['height']);factor=s
    raw=inp['global_joint_positions'];z=float(raw[:,[10,11],2].min());z=z-.1 if z>=.1 else z
    target=(raw-[0,0,z])*s;human=data['human_joints'];np.testing.assert_allclose(human,target,atol=2e-6)
    expected=inp['object_poses'].copy();z0=float(expected[0,6])
    expected[:,4:6]*=s;expected[:,6]=z0+(expected[:,6]-z0)*s
    np.testing.assert_allclose(q[:,36:39],expected[:,4:],atol=2e-6)
    np.testing.assert_allclose(q[:,39:43],expected[:,:4],atol=2e-6)
    extra=dict(object_error_translation_mm=float(np.linalg.norm(q[:,36:39]-expected[:,4:],axis=1).mean()*1000),object_error_kind='prescribed native-height trajectory with scaled mesh',human_floor_offset_m=z,source_object_initial_z_m=z0)
  assert q.shape==(frames,43) and np.isfinite(q).all()
  state=scene(obj,factor);legacy.scene=lambda unused:state
  metrics=legacy.evaluate(q,obj,src);metrics.update(joint_error(q,state[0],human));metrics.update(extra)
  row[method]=metrics;row[method+'_scale']=s;row[method+'_shared_mesh_scale']=factor
 write(OUT/'per_sequence'/f'{seq}.json',row)
 return row

def report():
 rows=[json.loads(p.read_text()) for p in sorted((OUT/'per_sequence').glob('*.json'))]
 paired=[r for r in rows if r['umr_ok'] and r['omni_ok']]
 summary=[]
 for obj in ['overall']+list(MANIFEST['objects']):
  chosen=rows if obj=='overall' else [r for r in rows if r['object']==obj]
  common=[r for r in chosen if r['umr_ok'] and r['omni_ok']]
  expected=90 if obj=='overall' else 15
  for method in ('umr','omni'):
   mean=lambda key:float(np.mean([r[method][key] for r in common])) if common else None
   summary.append(dict(object=obj,method=method,attempted=len(chosen),expected=expected,completed=sum(r[method+'_ok'] for r in chosen),paired_n=len(common),success_rate_percent=(100*sum(r[method+'_ok'] for r in chosen)/expected if len(chosen)==expected else None),obj_penetration_mm=mean('obj_pen_p95_mm'),ground_penetration_mm=mean('gnd_pen_p95_mm'),foot_skating_cm_s=None if not common else mean('skate_ms')*100,contact_recall_percent=None if not common else mean('contact_rec')*100,contact_f1_percent=None if not common else mean('contact_f1')*100,joint_error_mm=mean('joint_error_mm')))
 write(OUT/'results.json',rows);write(OUT/'summary.json',summary)
 with (OUT/'summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 flat=[]
 for row in rows:
  for method in ('umr','omni'):
   flat.append(dict(sequence=row['sequence'],object=row['object'],method=method,success=row[method+'_ok'],**row.get(method,{})))
 with (OUT/'per_sequence.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in flat for k in r)));w.writeheader();w.writerows(flat)
 lines=['# UMR versus OmniRetarget: OMOMO with scaled objects and native Omni height','',f"Status: {'complete' if len(rows)==90 else 'IN PROGRESS'} — {len(rows)}/90 sequence pairs evaluated. {len(paired)} common successful clips.",'','90 fixed clips, 15 per object, 30 FPS. UMR solver code is unchanged from upstream fcdce48. OmniRetarget uses the SMPL-X HOI extension, a uniformly scaled object mesh and the original OmniRetarget first-frame object-height rule. All 90 OmniRetarget solves are fresh; the 90 validated UMR outputs are reused byte-for-byte. Existing source evaluation labels are reused.','', 'Success means a complete kinematic solve. Quality metrics average each clip equally over common successes. Success rates are shown only after every clip in that cohort has been attempted; pending entries are marked —.','', '| Object | Method | Completed | Paired n | Success Rate (%) ↑ | Obj Penetration (mm) ↓ | Ground Penetration (mm) ↓ | Foot Skating (cm/s) ↓ | Contact Recall (%) ↑ | Contact F1 (%) ↑ | Joint Error (mm) ↓ |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in summary:
  fmt=lambda key:'—' if r[key] is None else f'{r[key]:.2f}'
  lines.append(f"| {r['object']} | {r['method'].upper()} | {r['completed']}/{r['expected']} | {r['paired_n']} | "+' | '.join(fmt(k) for k in ['success_rate_percent','obj_penetration_mm','ground_penetration_mm','foot_skating_cm_s','contact_recall_percent','contact_f1_percent','joint_error_mm'])+' |')
 lines+=['','Penetration is the per-clip 95th percentile of deepest robot penetration per frame, averaged across paired clips. Contact uses the existing 20 mm threshold and union of the two hands. No-source-contact clips retain recall/F1=0. Joint error is pelvis-relative position error across the same 14 mapped landmarks. See PROTOCOL.md for the exact native height rule, source labels and paired evaluation geometry.','','## Contact-active paired subset','']
 active=[r for r in paired if r['umr']['src_contact_frames']>0]
 for m in ('umr','omni'):
  if active:lines.append(f"- {m.upper()}: n={len(active)}, recall={np.mean([r[m]['contact_rec'] for r in active])*100:.2f}%, F1={np.mean([r[m]['contact_f1'] for r in active])*100:.2f}%.")
 failures=[f"- {r['sequence']} / {m}: {r[m+'_attempt']['reason']}" for r in rows for m in ('umr','omni') if not r[m+'_ok']]
 lines+=['','## Failed attempts','']+(failures or ['None among the evaluated attempts.'])
 tmp=OUT/'REPORT.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(OUT/'REPORT.md')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sequence');a=p.parse_args()
 if a.sequence:run(a.sequence)
 report()
