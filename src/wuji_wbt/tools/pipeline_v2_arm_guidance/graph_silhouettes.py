"""Overlay saved interaction graphs on the actual human and object surfaces.

Visualization only: consumes archived graph arrays and human mesh caches. It does
not construct new optimization graphs, change targets or run any retargeter.
Standalone HTML adds hover labels to the exact projected graph nodes, offline.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
import html
import sys
import time
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial.transform import Rotation
from config import *

BODY=['Pelvis','L_Hip','R_Hip','L_Knee','R_Knee','L_Shoulder','R_Shoulder',
      'L_Elbow','R_Elbow','L_Ankle','R_Ankle','L_Toe','R_Toe','L_Wrist','R_Wrist']
FINGERS=['Thumb','Index','Middle','Ring','Little']
HAND=['Wrist / palm']+[f'{finger} {part}' for finger in FINGERS for part in ('MCP','PIP','DIP','TIP')]
COLORS=['#d83f42','#e69a00','#4f963c','#3b7cbf','#9855b5']
BODY_BONES=[(0,1),(0,2),(1,3),(2,4),(3,9),(4,10),(9,11),(10,12),
            (0,5),(0,6),(5,7),(6,8),(7,13),(8,14)]


def surface(ax, vertices, faces, color, alpha, limits):
    triangles=vertices[faces]
    lo,hi=limits
    # Crop entire triangles conservatively; no geometry is decimated or resampled.
    keep=np.all(triangles.max(1)>=lo,axis=1)&np.all(triangles.min(1)<=hi,axis=1)
    collection=Poly3DCollection(triangles[keep],facecolor=color,edgecolor='none',
        linewidths=0,alpha=alpha,zsort='average',zorder=0)
    ax.add_collection3d(collection)


def panel(ax,g,kind,label,body_mesh,obj_mesh,rotation,position,zoom=False):
    vertices=g[kind]@rotation.T+position;n=g['n']
    combined=np.vstack([g['source'][:n] if zoom else g['source'],g['robot'][:n] if zoom else g['robot']])@rotation.T+position
    if n==15:combined=np.vstack([combined,body_mesh[0]])
    lo=combined.min(0);hi=combined.max(0);center=(lo+hi)/2
    extent=max(float((hi-lo).max())*.58,.075)
    limits=(center-extent,center+extent)
    for axis in range(3):getattr(ax,('set_xlim','set_ylim','set_zlim')[axis])(limits[0][axis],limits[1][axis])
    surface(ax,*obj_mesh,'#a48a69',.23,limits)
    surface(ax,*body_mesh,'#5a9bb8',.20 if n==15 else .26,limits)
    for k,color,alpha in ((2,'#8a8a8a',.10),(0,'#4879a3',.22),(1,'#dc8a2b',.40)):
        mask=g['kind']==k;lines=vertices[g['edges'][mask]]
        ax.add_collection3d(Line3DCollection(lines,colors=color,
            linewidths=.12+1.5*np.sqrt(g['weights'][mask]),alpha=alpha,zorder=2))
    ax.scatter(*vertices[n:].T,s=7,c='#81766a',alpha=.65,depthshade=False,zorder=3)
    if n==15:
        ax.add_collection3d(Line3DCollection(vertices[np.array(BODY_BONES)],colors='#16384a',linewidths=2.2,zorder=4))
        ax.scatter(*vertices[:n].T,s=24,c='#19364b',edgecolors='white',linewidths=.4,depthshade=False,zorder=5)
        for i in (0,5,6,7,8,13,14):
            ax.text(*vertices[i],BODY[i].replace('_',' '),fontsize=7,zorder=6,
                bbox=dict(facecolor='white',alpha=.75,edgecolor='none',pad=.5))
    else:
        ax.scatter(*vertices[0],s=30,c='#1c3446',depthshade=False,zorder=5)
        for f,color in enumerate(COLORS):
            chain=[0]+list(range(1+4*f,5+4*f))
            ax.plot(*vertices[chain].T,c=color,lw=2.2,zorder=4)
            ax.scatter(*vertices[chain[1:]].T,s=25,c=color,edgecolors='white',linewidths=.3,depthshade=False,zorder=5)
            i=4+4*f
            ax.text(*vertices[i],FINGERS[f] if zoom else FINGERS[f][0],color=color,fontsize=8,zorder=6,
                bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.5))
    ax.set_box_aspect((1,1,1));ax.view_init(elev=20,azim=-60)
    ax.set_xlabel('world x (m)',fontsize=7);ax.set_ylabel('world y (m)',fontsize=7);ax.set_zlabel('world z (m)',fontsize=7)
    ax.tick_params(labelsize=6);ax.grid(False)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.pane.fill=False
    ax.set_title(label,fontsize=11)
    names=BODY if n==15 else HAND
    return (ax,vertices,[f'{label.split(chr(10))[0]} | {names[i] if i<n else "Object sample "+str(i-n)} | graph node {i}' for i in range(len(vertices))],n)


def save_figure(fig,path,hovers,caption):
    fig.set_dpi(150);fig.canvas.draw()
    width,height=fig.canvas.get_width_height();fig.savefig(path,dpi=150)
    points=[]
    for ax,vertices,names,n in hovers:
        x,y,z=proj3d.proj_transform(*vertices.T,ax.get_proj())
        pixels=ax.transData.transform(np.c_[x,y]);bounds=ax.bbox
        limits=np.array([ax.get_xlim(),ax.get_ylim(),ax.get_zlim()])
        # Object hits first, anatomical hits last, to prioritize anatomical labels.
        for i in list(range(n,len(vertices)))+list(range(n)):
            px,py=pixels[i]
            if not bounds.contains(px,py) or np.any(vertices[i]<limits[:,0]) or np.any(vertices[i]>limits[:,1]):continue
            text=names[i]+'; world xyz (m): '+', '.join(f'{a:.4f}' for a in vertices[i])
            points.append(f'<circle class="hit" cx="{px:.3f}" cy="{height-py:.3f}" r="{5 if i>=n else 8}" data-info="{html.escape(text,quote=True)}"><title>{html.escape(text)}</title></circle>')
    viewer=path.with_name(path.stem+'_hover.html')
    viewer.write_text('''<!doctype html><meta charset="utf-8"><title>Interaction graph: labeled silhouettes</title>
<style>body{font:16px system-ui;margin:20px;background:#f7f9fb;color:#173246}svg{width:100%;height:auto;background:white}#info{position:sticky;top:0;padding:14px;background:#e6f1f8;border:1px solid #aac5d7;z-index:9}.hit{fill:transparent;stroke:transparent;pointer-events:all}.hit:hover{fill:#ffffff66;stroke:#132d43;stroke-width:2}</style>
'''+f'<h1>Interaction mesh over human and object silhouettes</h1><p>{html.escape(caption)}</p><p>Hover over a dot for its anatomical name or object-sample index. Thick colored lines are anatomical guides, not additional optimization constraints. Blue surface: actual human demonstration; tan surface: original object. The robot row uses the same human reference as a ghost, not a robot skin.</p><div id="info">Hover over a graph node.</div><svg viewBox="0 0 {width} {height}"><image href="{path.name}" width="{width}" height="{height}"/>'+''.join(points)+'</svg>'+'''
<script>for(const n of document.querySelectorAll('.hit')){n.addEventListener('mouseenter',()=>{document.getElementById('info').textContent=n.dataset.info});n.addEventListener('click',()=>{document.getElementById('info').textContent=n.dataset.info})}</script>''')
    plt.close(fig)
    return len(points)


def main(seq):
    started=time.perf_counter();info=inputs(seq);folder=OUT/'graphs'/seq;metadata=read(folder/'graphs.json')
    z=np.load(info['canonical']);poses=z['object_pose_mujoco']
    human=np.load(PILOT/'human'/f'{seq}.npz');human_vertices=human['vertices'];human_faces=human['faces']
    obj=trimesh.load(RET/'models'/info['object']/f"{info['object']}.obj",force='mesh');obj.apply_scale(float(z['scale']))
    records=[]
    for row in metadata['frames']:
        t=row['frame'];path=folder/f'frame_{t:03d}_graph.npz';before=sha(path);data=np.load(path)
        rotation=Rotation.from_quat(poses[t,[4,5,6,3]]).as_matrix();position=poses[t,:3]
        body_mesh=(human_vertices[t],human_faces);obj_mesh=(obj.vertices@rotation.T+position,obj.faces)
        assert np.isfinite(body_mesh[0]).all() and np.isfinite(obj_mesh[0]).all()
        graphs=[{k:data[f'{column}_{k}'] for k in ('source','robot','edges','weights','kind')} for column in range(3)]
        for column,g in enumerate(graphs):g['n']=15 if column==0 else 21
        # Confirm the hand nodes are exactly the same source-frame landmarks as the surface cache's scene.
        for side in range(2):assert np.allclose(graphs[side+1]['source'][:21]@rotation.T+position,z['hand_kp_scaled'][t,side],atol=1e-9)
        fig=plt.figure(figsize=(18,11.5));hovers=[]
        for col,g in enumerate(graphs):
            for r,kind in enumerate(('source','robot')):
                title=('Body','Left hand','Right hand')[col]+(' source graph' if r==0 else ' robot graph')
                sub='Human surface + object' if r==0 else 'Human reference ghost + object'
                ax=fig.add_subplot(2,3,1+col+3*r,projection='3d',computed_zorder=False)
                hovers.append(panel(ax,g,kind,title+'\n'+sub,body_mesh,obj_mesh,rotation,position))
        caption=f'{seq} | frame {t} ({t/30:.2f}s)'
        fig.suptitle(caption+'\nBlue translucent surface: human; tan: object. Orange: graph links to object samples.\nThick dark/colored lines show anatomy only. Bottom row: robot graph over the human reference.',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.92))
        count=save_figure(fig,folder/f'frame_{t:03d}_silhouettes.png',hovers,caption)
        fig=plt.figure(figsize=(13,11));hovers=[]
        for side,g in enumerate(graphs[1:]):
            for r,kind in enumerate(('source','robot')):
                ax=fig.add_subplot(2,2,1+side+2*r,projection='3d',computed_zorder=False)
                title=('Left','Right')[side]+f' hand {kind} graph\nHuman hand reference + object surface'
                hovers.append(panel(ax,g,kind,title,body_mesh,obj_mesh,rotation,position,zoom=True))
        fig.suptitle(caption+'\nHand close-up: actual human skin and object surface; each finger has a separate color.',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.94))
        count+=save_figure(fig,folder/f'frame_{t:03d}_hands_silhouettes.png',hovers,caption)
        assert sha(path)==before
        records.append(dict(frame=t,graph_sha256_unchanged=before,hover_points=count,
            human_triangles=len(human_faces),object_triangles=len(obj.faces),
            body_png=f'frame_{t:03d}_silhouettes.png',hand_png=f'frame_{t:03d}_hands_silhouettes.png'))
    write(folder/'silhouettes.json',dict(sequence=seq,frames=records,seconds=time.perf_counter()-started,
        graph_arrays_unchanged=True,optimizer_run=False,geometry='Original object triangle mesh and cached scaled human mesh, exact saved frame. Human mesh is also a reference ghost in robot-graph panels.',
        hand_color_order=dict(zip(FINGERS,COLORS)),anatomy_lines='Display guide only, not extra optimization edges'))
    print('SILHOUETTES',seq,round(time.perf_counter()-started,1),flush=True)


if __name__=='__main__':
    for seq in sys.argv[1:] or COHORT:main(seq)
