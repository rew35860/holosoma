"""Identify pairs responsible for threshold violations in the completed v2 pilot."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import mujoco

WORK=Path(__file__).resolve().parents[4]
OUT=WORK.parent/'holosoma/eval_out/ref_candidates/pipeline_v2_pilot_20261018'
RET=WORK/'src/holosoma_retargeting/holosoma_retargeting'
sys.path.insert(0,str(RET/'retarget_pipeline'))
from pipeline_v2 import CoupledRefiner, structurally_adjacent

result=[]
for seq in json.loads((OUT/'cohort.json').read_text())['sequences']:
    folder=OUT/'clips'/seq
    if not (folder/'per_frame_metrics.npz').exists():continue
    raw=np.load(folder/'per_frame_metrics.npz')
    obj=np.maximum(raw['hand_object_depth_m'].max(1),raw['body_object_depth'])
    ids=np.flatnonzero((obj>.02)|(raw['ground_depth']>.01)|(raw['self_depth']>.01))
    if not len(ids):
        result.append(dict(sequence=seq,violating_frames=0,pairs=[]));continue
    run=json.loads((folder/'run.json').read_text())
    metrics=json.loads((folder/'metrics.json').read_text())
    m=mujoco.MjModel.from_xml_path(run['model']);d=mujoco.MjData(m)
    q=np.load(folder/'final_v2.npz')['qpos']
    r=CoupledRefiner(m,metrics['object'])
    moving=set(int(m.jnt_bodyid[j]) for j in r.joints)
    def name(g):
        return dict(geom=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g),
                    body=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])))
    def can_move(g):
        b=int(m.geom_bodyid[g])
        while b:
            if b in moving:return True
            b=int(m.body_parentid[b])
        return False
    active=[g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    objects=[g for g in active if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '').startswith(metrics['object']+'_piece')]
    planes=[g for g in active if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE]
    robot=[g for g in active if g not in set(objects+planes)]
    robotset=set(robot)
    pairs=[]
    for t in ids:
        d.qpos[:]=q[t];mujoco.mj_forward(m,d)
        events=[]
        for a in robot:
            for b in objects+planes:
                threshold=.01 if b in planes else .02
                if b not in planes and np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b])>m.geom_rbound[a]+m.geom_rbound[b]:continue
                distance=float(mujoco.mj_geomDistance(m,d,a,b,.05,None))
                if distance < -threshold:events.append((a,b,-distance,'ground' if b in planes else 'object'))
        for c in d.contact:
            a,b=int(c.geom1),int(c.geom2)
            if a in robotset and b in robotset and c.dist<-.01 and not structurally_adjacent(m,int(m.geom_bodyid[a]),int(m.geom_bodyid[b])):
                events.append((a,b,float(-c.dist),'self'))
        pairs.append(dict(frame=int(t),violations=[dict(a=name(a),b=name(b),depth_mm=depth*1000,
                          kind=kind,at_least_one_moving_chain=can_move(a) or can_move(b)) for a,b,depth,kind in events]))
    result.append(dict(sequence=seq,violating_frames=len(ids),pairs=pairs))
(OUT/'residual_collisions.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
