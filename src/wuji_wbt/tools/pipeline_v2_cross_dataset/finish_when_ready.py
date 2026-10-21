"""Detached follow-through so validation and videos survive a closed session."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import time,subprocess
from config import *

def active(unit):
    result=subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True)
    return result.stdout.strip() in ('active','activating','deactivating')

if __name__=='__main__':
    while True:
        running=active('v2-cross-dataset-20261021.service')
        helpers=any(active(unit) for unit in ('v2-cross-organizer-recovery-20261021.service','v2-cross-bucket-tuning-20261021.service'))
        if not running and not helpers:
            assert (OUT/'current_outcomes.json').exists() and (OUT/'tuning_outcomes.json').exists(), 'Batch ended before all phases; inspect worker_failure.json and service logs'
            break
        time.sleep(20)
    import finish
    finish.main()
