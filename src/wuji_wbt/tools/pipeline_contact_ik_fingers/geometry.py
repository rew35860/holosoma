"""Independent Coal convex queries using the unchanged compiled MuJoCo shapes."""
import numpy as np
import mujoco
import coal


class ConvexQueries:
    def __init__(self,m):
        self.m=m;self.shapes={};cache={}
        for g in np.flatnonzero((m.geom_contype!=0)|(m.geom_conaffinity!=0)):
            kind=int(m.geom_type[g]);size=m.geom_size[g]
            if kind==mujoco.mjtGeom.mjGEOM_MESH:
                mid=int(m.geom_dataid[g])
                if mid not in cache:
                    start=int(m.mesh_vertadr[mid]);count=int(m.mesh_vertnum[mid]);points=coal.StdVec_Vec3s()
                    for v in m.mesh_vert[start:start+count]:points.append(v.astype(float))
                    cache[mid]=coal.Convex.convexHull(points,True,'Qt')
                shape=cache[mid]
            elif kind==mujoco.mjtGeom.mjGEOM_BOX:shape=coal.Box(2*size)
            elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:shape=coal.Sphere(float(size[0]))
            elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:shape=coal.Capsule(float(size[0]),float(2*size[1]))
            elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:shape=coal.Cylinder(float(size[0]),float(2*size[1]))
            elif kind==mujoco.mjtGeom.mjGEOM_ELLIPSOID:shape=coal.Ellipsoid(size)
            elif kind==mujoco.mjtGeom.mjGEOM_PLANE:shape=None
            else:raise ValueError(f'Unsupported collision geometry {kind}')
            self.shapes[int(g)]=shape
        self.request=coal.DistanceRequest(True,0.,0.)
        self.request.enable_signed_distance=True
        self.request.gjk_tolerance=1e-9;self.request.epa_tolerance=1e-9
        self.request.gjk_max_iterations=256;self.request.epa_max_iterations=256

    def query(self,d,a,b,cutoff=.04):
        a,b=int(a),int(b)
        if self.shapes[a] is None or self.shapes[b] is None:
            witness=np.zeros(6)
            distance=float(mujoco.mj_geomDistance(self.m,d,a,b,cutoff,witness))
            delta=witness[:3]-witness[3:];norm=np.linalg.norm(delta)
            normal=np.sign(distance)*delta/max(norm,1e-12)
            return distance,witness[:3].copy(),witness[3:].copy(),normal
        Ta=coal.Transform3s(d.geom_xmat[a].reshape(3,3),d.geom_xpos[a])
        Tb=coal.Transform3s(d.geom_xmat[b].reshape(3,3),d.geom_xpos[b])
        result=coal.DistanceResult()
        distance=float(coal.distance(self.shapes[a],Ta,self.shapes[b],Tb,self.request,result))
        pa=np.array(result.getNearestPoint1());pb=np.array(result.getNearestPoint2())
        normal=-np.asarray(result.normal)
        if not np.isfinite(distance) or not np.isfinite(normal).all():raise RuntimeError('Nonfinite convex query')
        return distance,pa,pb,normal
