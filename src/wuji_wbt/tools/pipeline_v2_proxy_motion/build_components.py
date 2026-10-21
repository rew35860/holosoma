"""Compact component decomposition; repair collinear T-junctions without moving vertices."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import time, json
import numpy as np
import trimesh, coacd
from config import *

def clean(mesh):
    mesh.merge_vertices(merge_tex=True,merge_norm=True)
    mesh.update_faces(mesh.unique_faces());mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh

def repair_t_junctions(mesh):
    """Split a boundary triangle edge at existing collinear boundary vertices."""
    before_area=float(mesh.area); changes=0
    for _ in range(1000):
        counts=np.bincount(mesh.edges_unique_inverse)
        boundary=mesh.edges_unique[counts==1]
        if not len(boundary):break
        ids=np.unique(boundary); points=mesh.vertices[ids]; found=False
        for a,b in boundary:
            edge=mesh.vertices[b]-mesh.vertices[a];den=float(edge@edge)
            if den<1e-20:continue
            fraction=(points-mesh.vertices[a])@edge/den
            distances=np.linalg.norm(points-(mesh.vertices[a]+fraction[:,None]*edge),axis=1)
            candidates=np.flatnonzero((fraction>1e-7)&(fraction<1-1e-7)&(distances<1e-9))
            if not len(candidates):continue
            vi=int(ids[candidates[np.argmin(fraction[candidates])]])
            fi=np.flatnonzero(np.any(mesh.faces==a,axis=1)&np.any(mesh.faces==b,axis=1))
            assert len(fi)==1
            triangle=mesh.faces[fi[0]].tolist()
            ia,ib=triangle.index(int(a)),triangle.index(int(b))
            if (ia+1)%3!=ib:a,b=b,a
            third=next(v for v in triangle if v not in (a,b))
            faces=np.delete(mesh.faces,fi[0],axis=0)
            mesh=trimesh.Trimesh(mesh.vertices,np.vstack([faces,[a,vi,third],[vi,b,third]]),process=False)
            changes+=1;found=True;break
        if not found:break
    assert abs(mesh.area-before_area)<1e-10, 'Repair must not change the source surface area'
    if mesh.is_watertight:
        trimesh.repair.fix_normals(mesh,multibody=True)
    return mesh,changes

def stool_wall_cells(mesh):
    """Clip original triangles into skirt/leg sectors before taking local hulls.

    Native stool axes are y-up, x-width, z-depth. The central cell is empty
    below the seat; retain that cavity. This never remeshes or offsets a face.
    """
    def clip(source,axis,low,high):
        value=source.copy();normal=np.eye(3)[axis]
        if np.isfinite(low):value=trimesh.intersections.slice_mesh_plane(value,normal,normal*low,cap=False)
        if np.isfinite(high) and len(value.faces):value=trimesh.intersections.slice_mesh_plane(value,-normal,normal*high,cap=False)
        return value
    groups=[]
    seat=clip(mesh,1,.103,np.inf)
    groups.append(('seat',seat.convex_hull,1))
    for band,(yl,yh) in enumerate(((-np.inf,-.045),(-.045,.103))):
        layer=clip(mesh,1,yl,yh)
        for ix,(xl,xh) in enumerate(((-np.inf,-.13),(-.13,.13),(.13,np.inf))):
            for iz,(zl,zh) in enumerate(((-np.inf,-.09),(-.09,.09),(.09,np.inf))):
                part=clip(clip(layer,0,xl,xh),2,zl,zh)
                if not len(part.faces) or part.area<1e-10:continue
                assert not(ix==1 and iz==1),'Unexpected material in central cavity'
                if np.linalg.matrix_rank(part.vertices-part.vertices.mean(0),tol=1e-9)<3:continue
                # A lower skirt cell can intersect two distinct legs. A single
                # hull across them would wrongly close the opening between them.
                if band==0:
                    part.merge_vertices();part.update_faces(part.nondegenerate_faces())
                    fragments=part.split(only_watertight=False)
                else:
                    fragments=[part]
                for k,fragment in enumerate(fragments):
                    assert np.linalg.matrix_rank(fragment.vertices-fragment.vertices.mean(0),tol=1e-9)==3
                    groups.append((f'wallcell_{band}_{ix}_{iz}_{k}',fragment.convex_hull,1))
    return groups

def main(obj):
    folder=EXP/'geometry/components'/obj;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'build.json').exists():return
    start=time.perf_counter();source=RET/'models'/obj/f'{obj}.obj'
    raw=clean(trimesh.load(source,force='mesh'));comps=raw.split(only_watertight=False)
    repairs=[];groups=[]
    if obj=='utility_cart':
        # Each wheel assembly is distant from the hand targets. Preserve outer bounds
        # with one hull; keep handles and all three tray cavities separate.
        for j,ids in enumerate(((0,1,2,3),(4,5,6,7),(10,11,12,13),(14,15,16,17))):
            groups.append((f'wheel_{j}',trimesh.util.concatenate([comps[i] for i in ids]).convex_hull,1))
        for i,cap in ((8,6),(9,6),(18,6),(19,5),(20,5)):
            groups.append((f'component_{i}',comps[i],cap))
    else:
        assert obj=='step_stool'
        for i,part in enumerate(comps):
            fixed,n=repair_t_junctions(part)
            repairs.append(dict(component=i,edge_splits=n,watertight=bool(fixed.is_watertight),
                winding_consistent=bool(fixed.is_winding_consistent),area_delta=float(fixed.area-part.area)))
            assert fixed.is_watertight and fixed.is_winding_consistent,(i,repairs[-1])
            if i==0:
                groups.extend(stool_wall_cells(fixed))
            else:
                groups.append((f'component_{i}',fixed.convex_hull,1))
    settings=dict(threshold=.01,max_convex_hull=32,mcts_iterations=200,resolution=2000,
        preprocess_mode='auto',preprocess_resolution=50,merge=True,extrude=False,seed=0)
    parts=[];records=[];coacd.set_log_level('info')
    for name,mesh,cap in groups:
        mesh.export(folder/f'{name}_source.obj')
        assert mesh.is_watertight and mesh.is_winding_consistent,name
        if mesh.volume<0:mesh.invert()
        begin=time.perf_counter()
        if cap==1:
            result=[mesh.convex_hull]
        else:
            kwargs=dict(settings,max_convex_hull=cap)
            result=[trimesh.Trimesh(v,f,process=False) for v,f in coacd.run_coacd(coacd.Mesh(mesh.vertices,mesh.faces),**kwargs)]
        records.append(dict(name=name,cap=cap,pieces=len(result),seconds=time.perf_counter()-begin,
            source_faces=len(mesh.faces),source_volume=float(mesh.volume),source_bounds=mesh.bounds.tolist()))
        parts.extend(result)
        print('GROUP_FINISHED',name,len(result),flush=True)
    assert len(parts)<=32
    for i,part in enumerate(parts):part.export(folder/f'piece_{i:03d}.obj')
    write(folder/'build.json',dict(object=obj,pieces=len(parts),seconds=time.perf_counter()-start,
        source=str(source),source_sha256=sha(source),groups=records,repairs=repairs,settings=settings,
        method='Physical components; collinear edge-splitting repair for stool; local convex hulls of original stool wall/leg sectors and seat, with hollow central cells excluded; convex wheel/rib groups; component-wise CoACD for cart handles and trays.',
        stool_cell_cuts_raw_m=dict(y=[-.045,.103],x=[-.13,.13],z=[-.09,.09]) if obj=='step_stool' else None,
        accuracy_claim='Candidate only: contact surface and opening error measured separately. Fixed per-component caps may exceed local concavity threshold.',production_assets_modified=False))

if __name__=='__main__':main(sys.argv[1])
