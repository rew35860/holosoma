"""Finite original-mesh patches and unsigned physical-surface separation.

Contact patches are not convexified. Robot tip geometry is its existing collision
convex. Negative material distances are handled separately by signed collisions.
"""
import numpy as np
import coal

def mesh_shape(triangles):
    triangles=np.asarray(triangles, float).reshape(-1,3,3)
    assert len(triangles)
    vertices=coal.StdVec_Vec3s();faces=coal.StdVec_Triangle()
    for point in triangles.reshape(-1,3):vertices.append(point)
    for i in range(len(triangles)):faces.append(coal.Triangle(3*i,3*i+1,3*i+2))
    shape=coal.BVHModelOBBRSS();shape.beginModel(len(faces),len(vertices))
    shape.addSubModel(vertices,faces);shape.endModel()
    return shape

def unsigned_query(a, Ta, b, Tb):
    req=coal.DistanceRequest(True,0.,0.);req.enable_signed_distance=False
    req.gjk_tolerance=1e-9;req.epa_tolerance=1e-9
    result=coal.DistanceResult()
    value=float(coal.distance(a,Ta,b,Tb,req,result))
    pa=np.asarray(result.getNearestPoint1()).copy()
    pb=np.asarray(result.getNearestPoint2()).copy()
    if value<=1e-12 and (not np.isfinite(pa).all() or not np.isfinite(pb).all()):
        # Coal's unsigned mesh/convex distance is zero on intersection but can
        # leave its nearest points undefined. Obtain a collision contact sample;
        # zero gap has zero residual derivative in the intersection interior.
        creq=coal.CollisionRequest();creq.num_max_contacts=1;creq.enable_contact=True
        cres=coal.CollisionResult();coal.collide(a,Ta,b,Tb,creq,cres)
        assert cres.numContacts()>0,'Zero distance with no finite witness or contact'
        pa=np.asarray(cres.getContact(0).pos).copy();pb=pa.copy()
    assert np.isfinite(value) and np.isfinite(pa).all() and np.isfinite(pb).all()
    return max(value,0.),pa,pb

def clip_triangles(triangles, center, radius):
    """Clip to a 40-mm cube so a large face cannot bypass the local region."""
    result=[]
    for tri in triangles:
        polygon=list(tri-center)
        for axis in range(3):
            for sign in (-1,1):
                output=[]
                if not polygon:break
                previous=polygon[-1];dp=sign*previous[axis]-radius
                for current in polygon:
                    dc=sign*current[axis]-radius
                    if (dc<=0)!=(dp<=0):
                        output.append(previous+(current-previous)*dp/(dp-dc))
                    if dc<=0:output.append(current)
                    previous=current;dp=dc
                polygon=output
        for i in range(1,len(polygon)-1):
            t=np.array([polygon[0],polygon[i],polygon[i+1]])+center
            if np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))>1e-14:result.append(t)
    return np.asarray(result).reshape(-1,3,3)

def object_patch(mesh, seed, anchor, adjacency, radius, cosine):
    triangles=mesh.triangles
    eligible=(np.all(triangles.max(1)>=anchor-radius,axis=1)&
              np.all(triangles.min(1)<=anchor+radius,axis=1)&
              (mesh.face_normals@mesh.face_normals[seed]>=cosine))
    visited={int(seed)};stack=[int(seed)]
    while stack:
        for nxt in adjacency[stack.pop()]:
            if eligible[nxt] and nxt not in visited:visited.add(nxt);stack.append(nxt)
    ids=np.asarray(sorted(visited))
    patch=clip_triangles(triangles[ids],anchor,radius)
    assert len(patch)>0
    return patch,ids
