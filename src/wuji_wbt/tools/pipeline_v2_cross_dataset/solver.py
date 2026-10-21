"""Shared current solver and initializer; explicit one-weight candidates only."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import time
import numpy as np
from config import *
backend=module('cross_backend',BACKEND_CODE/'solver.py')
load,target_at,mujoco=backend.load,backend.target_at,backend.mujoco
from holosoma_retargeting.retarget_pipeline.collision_aware_initialization import initialize_collision_aware
from holosoma_retargeting.retarget_pipeline import v3_preset

def main(seq):
    start=time.perf_counter();info,z,data,ref,body,init=load(seq,'mujoco')
    ref.queries=backend.MujocoQueries(ref.queries)
    dest=destination(seq);dest.mkdir(parents=True,exist_ok=True)
    if RUN_VARIANT=='tip_relaxed':ref.cfg.fingertip_weight=6.25
    if RUN_VARIANT==v3_preset.EXPERIMENT_VARIANT:v3_preset.apply_weights(ref,EXTRA)
    begin=time.perf_counter();target=target_at(ref,z,data,0)
    q,placement=initialize_collision_aware(ref,target,body[0],init[0],z['hand_kp_scaled'][0])
    first_seconds=time.perf_counter()-begin;write(dest/'initial_placement.json',placement)
    Q=body.copy();Q[0]=q;previous=q.copy();records=[];begin=time.perf_counter()
    for t in range(1,len(Q)):
        q=init[t].copy();q[ref.qadr]=previous[ref.qadr]
        q,predictor=ref.compensate(q,previous,data['contact'][t],data['contact'][t-1],1.)
        q,record=ref.solve_stage(q,target_at(ref,z,data,t),body[t],init[t],previous,first=False)
        record.update(frame=t,predictor=predictor);records.append(record);Q[t]=q;previous=q.copy()
        if t%30==0 or t==len(Q)-1:
            print(seq,RUN_VARIANT,t+1,len(Q),flush=True)
            write(dest/'status.json',dict(state='solving',frames=t+1,total=len(Q)))
    elapsed=time.perf_counter()-begin
    fixed=np.setdiff1d(np.arange(ref.m.nq),ref.qadr);np.testing.assert_array_equal(Q[:,fixed],body[:,fixed])
    assert np.isfinite(Q).all() and np.all(Q[:,ref.qadr]>=ref.lo-1e-7) and np.all(Q[:,ref.qadr]<=ref.hi+1e-7)
    assert np.abs(np.diff(Q[:,ref.qadr],axis=0)).max()<=ref.cfg.joint_rate_limit+1e-6
    np.savez_compressed(dest/'final_v2.npz',qpos=Q,fps=30,human_joints=z['human_joints_scaled'])
    write(dest/'iterations.json',records)
    write(dest/'run.json',dict(sequence=seq,variant=RUN_VARIANT,pipeline_version='v3' if RUN_VARIANT==v3_preset.EXPERIMENT_VARIANT else 'v2',frames=len(Q),settings=vars(ref.cfg),extra=EXTRA,options=OPTIONS,
        initializer_seconds=first_seconds,subsequent_refinement_seconds=elapsed,joint_refinement_seconds=first_seconds+elapsed,
        total_seconds=time.perf_counter()-start,fixed_body_object_exact=True,signed_recovery=False,
        geometry_side=True,motion_compensated_init=True,substeps=False,
        final_sha256=sha(dest/'final_v2.npz'),evaluation_targets_sha256=sha(OUT/'targets'/f'{seq}.npz')))
    write(dest/'status.json',dict(state='solved',frames=len(Q)))
if __name__=='__main__':run_stage('solver',main,sys.argv[1])
