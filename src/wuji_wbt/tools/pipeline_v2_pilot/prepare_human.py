"""Skin HUMOTO's own source human mesh on CPU; verify it matches canonical hand targets."""
from pathlib import Path
import hashlib
import json
import pickle
import sys
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from catalog import ROOT, OUT, baseline_for, source_pkl
sys.path.insert(0,str(ROOT/'Data/humoto'))
from human_model.human_model import HumanModelDifferentiable

def prepare(seq):
    start=time.perf_counter();folder=OUT/'human';folder.mkdir(exist_ok=True)
    output=folder/f'{seq}.npz'
    if output.exists() and output.with_suffix('.json').exists():return output
    source=source_pkl(seq)
    with source.open('rb') as f:data=pickle.load(f)
    arm=data['armature']
    canonical=baseline_for(seq)/'clips'/seq/'part0'/f'{seq}_canonical.npz'
    z=np.load(canonical);target=z['hand_kp_scaled'];scale=float(z['scale'])
    assert len(arm)==len(target),(seq,len(arm),len(target))
    modelpath=ROOT/'Data/humoto/human_model/human_model_up_bone_zup.json'
    model=HumanModelDifferentiable(str(modelpath),device='cpu')
    handnames=[[f'mixamorig:{side}Hand']+[f'mixamorig:{side}Hand{finger}{j}'
                for finger in ('Thumb','Index','Middle','Ring','Pinky') for j in (1,2,3,4)]
               for side in ('Left','Right')]
    verts=[];hands=[];hips=[]
    with torch.no_grad():
        for startframe in range(0,len(arm),16):
            chunk=arm[startframe:startframe+16];poses={}
            for name in chunk[0]:
                a=np.asarray([f[name] for f in chunk],np.float32)
                mats=np.tile(np.eye(4,dtype=np.float32),(len(a),1,1))
                mats[:,:3,:3]=Rotation.from_quat(a[:,[1,2,3,0]]).as_matrix()
                mats[:,:3,3]=a[:,4:7];poses[name]=torch.from_numpy(mats)
            v,joints=model(poses)
            verts.append(v.numpy().copy());hips.append(joints['mixamorig:Hips'].numpy().copy())
            hands.append(np.stack([np.stack([joints[n].numpy() for n in ns],axis=1) for ns in handnames],axis=1))
    vertices=np.concatenate(verts);human_hands=np.concatenate(hands);pelvis=np.concatenate(hips)
    # One constant scene translation accounts for the existing packing floor shift.
    # No per-frame alignment or pose adjustment is permitted.
    translation=z['human_joints_scaled'][0,0]-scale*pelvis[0]
    canonical_hands=human_hands*scale+translation
    err=float(np.max(np.abs(canonical_hands-target)))
    hiperr=float(np.max(np.abs(pelvis*scale+translation-z['human_joints_scaled'][:,0])))
    assert err<2e-5 and hiperr<2e-5,(seq,err,hiperr)
    vertices=vertices*scale+translation
    np.savez_compressed(output,vertices=vertices.astype(np.float32),faces=model.triangulated_faces.astype(np.int32),fps=30)
    metadata=dict(sequence=seq,frames=len(vertices),vertices=vertices.shape[1],faces=len(model.triangulated_faces),
        source_pickle=str(source),source_model=str(modelpath),canonical=str(canonical),
        source_pickle_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scale=scale,
        constant_scene_translation_m=translation.tolist(),hand_target_max_coordinate_error_m=err,
        pelvis_target_max_coordinate_error_m=hiperr,seconds=time.perf_counter()-start,
        source='Original HUMOTO standardized up-bone human skin; no SMPL-X fit, no hand optimization; same scene scaling as the retargeter')
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata),flush=True);return output

if __name__=='__main__':
    for seq in sys.argv[1:]:prepare(seq)
