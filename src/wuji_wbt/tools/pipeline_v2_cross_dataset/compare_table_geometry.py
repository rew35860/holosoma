"""Score the old and new table motion with one common, surface-validated model."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
from config import *
import solver

def main():
    seq='push_table_forward_with_both_hands-039';info,z,data,ref,body,init=solver.load(seq,'mujoco')
    old=OUT/'geometry_preflight/table_grail/variants/current/clips'/seq
    new=OUT/'variants/current/clips'/seq;results={}
    E=module('table_scores',BASE_CODE/'evaluate.py')
    for label,folder in [('coarse18',old),('components9',new)]:
        Q=np.load(folder/'final_v2.npz')['qpos'];pred=np.zeros(data['contact'].shape,bool);depth=[]
        for t,q in enumerate(Q):
            distances,_,ids=ref.collisions(q,False)
            obj=ref.pairs[ids,2]==0;depth.append(max(0.,-distances[obj].min(initial=0.)))
            for s in (0,1):
                for f,g in enumerate(ref.tipgeom[s]):
                    mask=obj&(ref.pairs[ids,0]==g);distance=distances[mask].min(initial=ref.cfg.collision_detect)
                    pred[t,s,f]=-.001<=distance<=.002
        d=read(folder/'diagnostics.json')
        results[label]=dict(physical_contact=E.scores(data['contact'],pred),
            registered_object_peak_mm=float(np.max(depth)*1000),original_mesh_patch_gap_mm=d['physical_patch_gap_mean_mm'])
    archived=np.load(OUT/'geometry_preflight/table_grail/targets'/f'{seq}.npz')
    targets_same=all(np.array_equal(value,archived[key],equal_nan=True) for key,value in data.items())
    assert targets_same
    write(OUT/'table_geometry_comparison.json',dict(results=results,
        common_evaluation_model=str(info['model']),common_collider_pieces=9,
        definitions='Same Coal distances, source labels and thresholds for both trajectories. Patch gap uses unchanged original mesh targets. This comparison avoids measuring each method with its own collision geometry.',
        source_targets_unchanged=targets_same))
if __name__=='__main__':main()
