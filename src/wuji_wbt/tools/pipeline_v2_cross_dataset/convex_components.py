"""Generic low-piece candidate: preserve each already-convex mesh component.

No object names, hand frames, or robot outcomes enter this decision. Both source
and hull surfaces are sampled; a failed component rejects the entire candidate.
This is a sampled bound, not a formal Hausdorff/cavity certificate.
"""
import numpy as np
import trimesh

def candidate(mesh,max_pieces=32,budget_m=.0005,seed=0):
    mesh=mesh.copy();mesh.merge_vertices(merge_tex=True,merge_norm=True)
    mesh.update_faces(mesh.unique_faces());mesh.update_faces(mesh.nondegenerate_faces())
    components=list(mesh.split(only_watertight=False));audit=dict(component_count=len(components),budget_m=budget_m,parts=[])
    if not 0<len(components)<=max_pieces:
        return None,dict(audit,accepted=False,reason='Connected-component count exceeds budget')
    hulls=[]
    for i,part in enumerate(components):
        try:
            hull=part.convex_hull
            if hull.volume<=1e-10:raise ValueError('Zero-thickness component has no supported solid interpretation')
            source=np.vstack([part.vertices,part.triangles_center,trimesh.sample.sample_surface(part,1000,seed=seed+i)[0]])
            target=np.vstack([hull.vertices,hull.triangles_center,trimesh.sample.sample_surface(hull,1000,seed=seed+i)[0]])
            a=trimesh.proximity.closest_point(hull,source)[1]
            b=trimesh.proximity.closest_point(part,target)[1]
            error=float(max(a.max(),b.max()))
        except Exception as exc:
            return None,dict(audit,accepted=False,reason=f'Component {i}: {exc}')
        audit['parts'].append(dict(component=i,source_faces=len(part.faces),hull_faces=len(hull.faces),
            source_to_hull_max_mm=float(a.max()*1000),hull_to_source_max_mm=float(b.max()*1000),sampled_error_mm=error*1000))
        if error>budget_m:return None,dict(audit,accepted=False,reason=f'Component {i} exceeds sampled surface budget')
        hulls.append(hull)
    return hulls,dict(audit,accepted=True,pieces=len(hulls),reason='Every connected component passes bidirectional sampled surface check')

if __name__=='__main__':
    import sys
    from config import OUT,write
    obj=sys.argv[1];folder=OUT/'geometry_component_candidates'/obj;folder.mkdir(parents=True,exist_ok=True)
    mesh=trimesh.load(OUT/'geometry'/obj/'visual.obj',force='mesh')
    parts,audit=candidate(mesh)
    if parts:
        for i,p in enumerate(parts):p.export(folder/f'piece_{i:03d}.obj')
    write(folder/'audit.json',audit);print(audit,flush=True)
