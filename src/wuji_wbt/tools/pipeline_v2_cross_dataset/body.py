"""Fresh body pass with required object colliders, unchanged body settings."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import time
import numpy as np
from config import *
sys.path.insert(0,str(RET.parent))
from holosoma_retargeting.retarget_pipeline.pipeline import PipelineConfig, MotionDataConfig, RobotConfig, TaskConfig, create_task_constants, part1_stageA
from holosoma_retargeting.retarget_pipeline.pipeline import InteractionMeshRetargeter

def main(seq):
    start=time.perf_counter(); info=inputs(seq); job=read(OUT/'jobs'/f'{seq}.json')
    cfg=PipelineConfig(task_name=seq,object_name=info['object'],stage='part1',save_dir=OUT/'clips'/seq,
        pen_tolerance=job['body_pen_tolerance'],stageA_model_base=Path(read(marker(seq,'canonical'))['unscaled_body_model']))
    motion=MotionDataConfig(data_format='smplh_wuji_body',robot_type='g1')
    constants=create_task_constants(robot_config=RobotConfig(robot_type='g1'),motion_data_config=motion,
        task_config=TaskConfig(object_name=info['object']),task_type='object_interaction')
    recovery=os.environ.get('PROXY_BODY_RECOVERY')=='1'
    history=[];events=[];original=InteractionMeshRetargeter.iterate
    def tracked(self,*args,**kwargs):
        frame=int(kwargs['frame_idx'])
        try:
            value=original(self,*args,**kwargs)
        except RuntimeError as error:
            entry=dict(frame=frame,original_error=str(error),attempts=[])
            events.append(entry)
            np.savez_compressed(OUT/'clips'/seq/'failed_body_frame.npz',
                q_n=kwargs['q_n'],q_t_last=kwargs['q_t_last'],q_locked=kwargs['q_locked'],frame=frame)
            value=None
            if recovery:
                # Same objective, constraints, previous-frame foot targets, trust
                # region and iteration count. Only change the nonlinear warm start.
                for dz in (.01,-.01,.03,-.03,.05,-.05,.10,-.10):
                    trial=dict(kwargs);trial['q_n']=kwargs['q_n'].copy();trial['q_n'][2]+=dz
                    try:
                        value=original(self,*args,**trial)
                        entry['attempts'].append(dict(root_z_seed_offset_m=dz,solved=True));break
                    except RuntimeError as exc:
                        entry['attempts'].append(dict(root_z_seed_offset_m=dz,solved=False,error=str(exc)))
            write(OUT/'clips'/seq/'body_retries.json',dict(policy='Shared root-z warm-start candidates +/-1,3,5,10cm; no objective/constraint/tolerance changes.',events=events))
            if value is None:
                np.savez_compressed(OUT/'clips'/seq/'body_partial.npz',qpos=np.array(history),fps=30)
                raise
        history.append(value[0].copy())
        if frame%25==0:np.savez_compressed(OUT/'clips'/seq/'body_progress.npz',qpos=np.array(history),fps=30)
        return value
    InteractionMeshRetargeter.iterate=tracked
    try:part1_stageA(cfg,motion,constants)
    finally:InteractionMeshRetargeter.iterate=original
    write(marker(seq,'body'),dict(seconds=time.perf_counter()-start,fresh_body=True,
        body_pen_tolerance=cfg.pen_tolerance,body_sha256=sha(info['body']),
        warm_start_recovery_enabled=recovery,recovered_frames=[e['frame'] for e in events]))

if __name__=='__main__':run_stage('body',main,sys.argv[1])
