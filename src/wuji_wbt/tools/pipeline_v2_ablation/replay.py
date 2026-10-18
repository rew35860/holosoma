"""Run the unchanged solver, recording failure locals only after an exception."""
import runpy
import sys
import traceback
import numpy as np
from config import SOLVER, write

args=sys.argv[1:]
dest=__import__('pathlib').Path(args[args.index('--output')+1])
sys.argv=[str(SOLVER),*args]
try:
    runpy.run_path(str(SOLVER),run_name='__main__')
except Exception as exc:
    diagnostic=dict(error=repr(exc),solver_modified=False)
    tb=exc.__traceback__
    while tb:
        frame=tb.tb_frame;loc=frame.f_locals
        if frame.f_code.co_name=='refine':
            diagnostic.update(frame_zero_based=int(loc['t']),sqp_iteration_zero_based=int(loc['it']))
            np.savez_compressed(dest/'failed_state.npz',qpos=loc['q'],previous=loc['previous'],
                human=loc['human'][loc['t']],object_pose=loc['pose'][loc['t']])
        if frame.f_code.co_name=='subproblem':
            diagnostic['clarabel_status']=str(loc['result'].status)
            diagnostic['active_collision_rows']=int(loc['k'])
            diagonal=loc['H'].diagonal()
            diagnostic['objective_diagonal_min_max']=[float(diagonal.min()),float(diagonal.max())]
            ev=np.linalg.eigvalsh(loc['H'])
            diagnostic['objective_hessian_eigenvalue_min_max']=[float(ev.min()),float(ev.max())]
            diagnostic['joint_rate_bound_intersection_min_width']=float(np.min(loc['high']-loc['low']))
        tb=tb.tb_next
    write(dest/'failure_diagnostic.json',diagnostic)
    traceback.print_exc()
    sys.exit(1)
