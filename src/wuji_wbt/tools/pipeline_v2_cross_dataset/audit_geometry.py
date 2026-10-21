"""Sampled target-region proxy error; never a global cavity certificate."""
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from config import *

def main(seq):
    info=inputs(seq);z=np.load(info['canonical']);d=np.load(OUT/'targets'/f'{seq}.npz')
    mask=d['contact'];anchors=d['anchor_local'][mask];directions=d['direction_local'][mask]
    ids=np.linspace(0,len(anchors)-1,min(500,len(anchors)),dtype=int);anchors=anchors[ids];directions=directions[ids]
    pieces=[]
    for p in sorted((OUT/'geometry'/info['object']).glob('piece_*.obj')):
        part=trimesh.load(p,force='mesh');part.apply_scale(float(z['scale']));pieces.append(part.convex_hull)
    eq=[ConvexHull(p.vertices).equations for p in pieces]
    sign=np.stack([np.max(anchors@e[:,:3].T+e[:,3],axis=1) for e in eq],axis=1).min(1)
    separation=np.full(len(anchors),np.inf)
    for part in pieces:
        lower=np.linalg.norm(np.maximum(np.maximum(part.bounds[0]-anchors,anchors-part.bounds[1]),0),axis=1)
        valid=np.flatnonzero(lower<separation)
        if len(valid):separation[valid]=np.minimum(separation[valid],trimesh.proximity.closest_point(part,anchors[valid])[1])
    error=np.where(sign<0,-sign,separation)
    write(OUT/'geometry_audit'/f'{seq}.json',dict(sequence=seq,object=info['object'],samples=len(anchors),pieces=len(pieces),
        anchor_absolute_error_mm=dict(zip(('median','p95','maximum'),np.percentile(error*1000,[50,95,100]).tolist())),
        anchor_burial_max_mm=float(np.maximum(-sign,0).max()*1000),above_1mm_pct=float(np.mean(error>.001)*100),
        passes_sampled_half_mm_budget=bool(error.max()<=.0005),
        definition='Signed convex halfspace burial (lower bound on union penetration) or nearest convex surface separation at up to 500 evenly spaced active source anchors. Original mesh anchors, same scale. Not all surfaces/openings certified.'))
if __name__=='__main__':main(sys.argv[1])
