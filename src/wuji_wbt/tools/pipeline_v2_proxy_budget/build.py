"""Isolated UMR/GRAIL-parameter CoACD candidate, using the current pipeline library."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys,time,json,hashlib,inspect,importlib.metadata
import trimesh,numpy as np,coacd

CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[4]
OUT=ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_proxy_budget_20261021'
AUDIT=ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_thin_proxy_audit_20261021'
PARAMS=dict(threshold=.03,max_convex_hull=32,mcts_iterations=200,resolution=2000,
            preprocess_resolution=50,preprocess_mode='auto',mcts_max_depth=3,mcts_nodes=20,
            pca=False,merge=True,decimate=False,extrude=False,seed=0,real_metric=False)

def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def main(obj):
    folder=OUT/obj;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'build.json').exists():return
    meta=json.loads((AUDIT/f'{obj}.json').read_text());source=Path(meta['source_path'])
    mesh=trimesh.load(source,force='mesh');coacd.set_log_level('info')
    write(folder/'status.json',dict(state='building',object=obj,parameters=PARAMS))
    start=time.perf_counter();cpu=time.process_time()
    parts=coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices,np.float64),np.asarray(mesh.faces,np.int32)),**PARAMS)
    seconds=time.perf_counter()-start;cpu_seconds=time.process_time()-cpu
    assert 0<len(parts)<=32
    for i,(v,f) in enumerate(parts):trimesh.Trimesh(v,f,process=False).export(folder/f'piece_{i:03d}.obj')
    write(folder/'build.json',dict(object=obj,pieces=len(parts),parameters=PARAMS,seconds=seconds,cpu_seconds=cpu_seconds,
        coacd_version=importlib.metadata.version('coacd'),trimesh_version=trimesh.__version__,
        source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='UMR/GRAIL parameter preset applied with current pipeline CoACD. Not a bitwise replay of a different UMR environment.',
        production_assets_modified=False))
    write(folder/'status.json',dict(state='built',object=obj,pieces=len(parts),seconds=seconds))

if __name__=='__main__':main(sys.argv[1])
