"""Reconstruct the actual source Delaunay graphs and audit local objective gradients.

Graphs use the saved scaled canonical input, original body output and unchanged V2.
No mesh resampling, optimization or collision-setting changes occur here.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
import sys
import subprocess
import numpy as np
import mujoco
import trimesh
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from config import *
sys.path.insert(0, str(RET/'retarget_pipeline'))
from pipeline_v2 import CoupledRefiner, Settings, TIPS, initial_body
from holosoma_retargeting.config_types.data_type import JOINTS_MAPPINGS, SMPLH_DEMO_JOINTS
from holosoma_retargeting.src.utils import calculate_laplacian_matrix
from holosoma_retargeting.wuji.topoRetarget import create_interaction_mesh, get_adjacency_list, exp_laplacian_matrix


def graph(points, n, exponential):
    _, tets = create_interaction_mesh(points)
    adjacency = get_adjacency_list(tets, len(points))
    L = exp_laplacian_matrix(points, adjacency, kappa=30.) if exponential else calculate_laplacian_matrix(points, adjacency)
    edges = np.array(sorted({tuple(sorted((i, int(j)))) for i, ns in enumerate(adjacency) for j in ns}))
    weights = np.array([max(abs(L[i,j]), abs(L[j,i])) for i,j in edges])
    kind = np.array([2 if i >= n else 1 if j >= n else 0 for i,j in edges])
    return dict(source=points, L=L, edges=edges, weights=weights, kind=kind, n=n)


def panel(ax, g, vertices, title, labels=None, zoom=False):
    vertices=vertices@g['display_rotation'].T+g['display_translation']
    colors = ['#32699c', '#db771c', '#a9adb2']
    for kind in (2, 0, 1):
        selected = g['kind'] == kind
        widths = .15+2.0*np.sqrt(g['weights'][selected])
        ax.add_collection3d(Line3DCollection(vertices[g['edges'][selected]], colors=colors[kind],
            linewidths=widths, alpha=.12 if kind==2 else .55))
    n=g['n']
    ax.scatter(*vertices[n:].T, s=7, c='#888888', depthshade=False)
    ax.scatter(*vertices[:n].T, s=25, c='#226491', depthshade=False)
    if labels:
        for i,label in labels.items():
            ax.text(*vertices[i], label, fontsize=7)
    combined=np.vstack([g['source'][:n] if zoom else g['source'],g['robot'][:n] if zoom else g['robot']])@g['display_rotation'].T+g['display_translation']
    lo=combined.min(0);hi=combined.max(0)
    extent=max((hi-lo).max()*.55, .08);center=(hi+lo)/2
    ax.set_xlim(center[0]-extent,center[0]+extent);ax.set_ylim(center[1]-extent,center[1]+extent);ax.set_zlim(center[2]-extent,center[2]+extent)
    ax.set_box_aspect((1,1,1));ax.view_init(elev=20,azim=-60)
    ax.set_xlabel('world x (m)',fontsize=7);ax.set_ylabel('world y (m)',fontsize=7);ax.set_zlabel('world z (m)',fontsize=7)
    ax.tick_params(labelsize=6);ax.set_title(title,fontsize=11)


def main(seq):
    import collision_audit
    for variant in REMOVALS:
        dest=OUT/'variants'/variant/'clips'/seq
        if (dest/'completed.json').exists() and not (dest/'collision_audit.json').exists():
            collision_audit.main(variant,seq)
        if not (dest/'completed.json').exists() and (dest/'refinement.log').exists():
            retry=OUT/'retries'/variant/seq
            if not (retry/'retry.json').exists():
                retry.mkdir(parents=True,exist_ok=True)
                with (retry/'retry.log').open('w') as log:
                    subprocess.run([PY,str(CODE/'retry_failure.py'),variant,seq],check=True,stdout=log,stderr=subprocess.STDOUT)
    info=inputs(seq);z=np.load(info['canonical']);O=z['object_points_scaled']
    human,pose=z['hand_kp_scaled'],z['object_pose_mujoco']
    rotations=Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix()
    mesh=trimesh.load(RET/'models'/info['object']/f"{info['object']}.obj",force='mesh');mesh.apply_scale(float(z['scale']))
    pq=trimesh.proximity.ProximityQuery(mesh)
    local=np.einsum('tij,thki->thkj',rotations,human[:,:,TIPS]-pose[:,None,None,:3])
    near=[];dist=[];flat=local.reshape(-1,3)
    for i in range(0,len(flat),16):
        a,b,_=pq.on_surface(flat[i:i+16]);near.append(a);dist.append(b)
    contact=np.concatenate(dist).reshape(local.shape[:-1])<.03
    surface=np.einsum('tij,thkj->thki',rotations,np.concatenate(near).reshape(local.shape))+pose[:,None,None,:3]
    ma=mujoco.MjModel.from_xml_path(str(info['body_model']));da=mujoco.MjData(ma)
    m=mujoco.MjModel.from_xml_path(str(info['model']));ref=CoupledRefiner(m,info['object'],Settings(**BASE_SETTINGS))
    bodyfile=np.load(info['body']);qb=bodyfile['qpos'];Qbody=initial_body(m,ma,qb)
    assert np.allclose(bodyfile['human_joints'],z['human_joints_scaled'],atol=1e-10)
    q=np.load(PILOT/'clips'/seq/'final_v2.npz')['qpos'];init=np.load(PILOT/'clips'/seq/'finger_initialization.npz')['qpos']
    mapping=JOINTS_MAPPINGS['smplh_wuji_body','g1'];names=list(mapping)
    indices=[SMPLH_DEMO_JOINTS.index(n) for n in names]
    bids=[mujoco.mj_name2id(ma,mujoco.mjtObj.mjOBJ_BODY,v) for v in mapping.values()]
    assert min(bids)>=0, 'Archived body model must contain every mapped landmark'
    target_dir=OUT/'graphs'/seq;target_dir.mkdir(parents=True,exist_ok=True)
    counts=contact.sum((1,2));ts=np.flatnonzero(counts==counts.max());chosen=int(ts[len(ts)//2])
    frames=[chosen]+([140] if info['object']=='organizer_medium' and chosen!=140 else [])
    metadata=[]
    for t in frames:
        R=rotations[t];center=pose[t,:3]
        source=(z['human_joints_scaled'][t,indices]-center)@R
        bg=graph(np.vstack([source,O]),15,False)
        da.qpos[:]=qb[t];mujoco.mj_forward(ma,da)
        bg['robot']=np.vstack([(da.xpos[bids]-center)@R,O])
        ref.fk(q[t]);points=ref.keypoints(False)
        graphs=[bg]
        for side in range(2):
            g=graph(np.vstack([(human[t,side]-center)@R,O]),21,True)
            g['robot']=np.vstack([(points[side]-center)@R,O]);graphs.append(g)
        fig=plt.figure(figsize=(17,10.5));records=[];arrays={}
        for column,(g,label) in enumerate(zip(graphs,['Body pass: 15 landmarks','Coupled pass: LEFT hand','Coupled pass: RIGHT hand'])):
            g['display_rotation']=R;g['display_translation']=center
            labels={i:n.replace('Shoulder','Sh').replace('Elbow','El').replace('Wrist','Wr') for i,n in enumerate(names) if any(k in n for k in ('Shoulder','Elbow','Wrist'))} if column==0 else {0:'palm',4:'thumb',8:'index',12:'middle',16:'ring',20:'little'}
            cost=float(np.sum((g['L']@(g['robot']-g['source']))**2))
            if column==0:weighted=10*cost
            else:
                rr=g['L']@(g['robot']-g['source']);rr[TIPS]*=np.sqrt(BASE_SETTINGS['tip_weight']);weighted=BASE_SETTINGS['lambda_im']/len(rr)*float(np.sum(rr**2))
            panel(fig.add_subplot(2,3,column+1,projection='3d'),g,g['source'],label+'\nHuman source graph',labels)
            panel(fig.add_subplot(2,3,column+4,projection='3d'),g,g['robot'],f'Same edges on saved robot pose\nweighted mesh cost = {weighted:.4g}',labels)
            records.append(dict(pass_name=label,nodes=len(g['source']),human_robot_nodes=g['n'],object_nodes=len(O),edges=len(g['edges']),cross_edges=int(np.sum(g['kind']==1)),weighted_mesh_cost=weighted))
            for k in ('source','robot','L','edges','weights','kind'):arrays[f'{column}_{k}']=g[k]
        fig.suptitle(f'{seq} | frame {t} ({t/30:.2f}s)\nOrange: body/hand-to-object edges; blue: body/hand edges; grey: object edges. Width reflects neighbour weight.\nSource topology is held fixed within each frame. Coupled graph has no shoulder/elbow nodes.',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.92));fig.savefig(target_dir/f'frame_{t:03d}.png',dpi=150);plt.close(fig)
        np.savez_compressed(target_dir/f'frame_{t:03d}_graph.npz',**arrays)
        fig=plt.figure(figsize=(11,9))
        for side,g in enumerate(graphs[1:]):
            for row,kind in enumerate(('source','robot')):
                panel(fig.add_subplot(2,2,1+side+2*row,projection='3d'),g,g[kind],
                    f'{("Left","Right")[side]} hand: {kind} (zoom)',{0:'palm',4:'thumb',8:'index',12:'middle',16:'ring',20:'little'},zoom=True)
        fig.suptitle('Same coupled graphs, cropped around the hands; distant object edges continue outside view.')
        fig.tight_layout();fig.savefig(target_dir/f'frame_{t:03d}_hands.png',dpi=150);plt.close(fig)
        # Matrix view exposes weights even when a dense graph is visually occluded.
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        for ax,g,label in zip(axes,graphs,['Body: uniform neighbours','Left hand: exp(-30 distance)','Right hand: exp(-30 distance)']):
            W=np.eye(len(g['L']))-g['L'];im=ax.imshow(W,cmap='magma',vmin=0,vmax=1)
            ax.axvline(g['n']-.5,c='cyan',lw=.7);ax.axhline(g['n']-.5,c='cyan',lw=.7)
            ax.set_title(label);ax.set_xlabel('neighbour node');ax.set_ylabel('row node')
        fig.colorbar(im,ax=axes.tolist(),label='directed neighbour weight',shrink=.8)
        fig.savefig(target_dir/f'frame_{t:03d}_weights.png',dpi=150,bbox_inches='tight');plt.close(fig)
        metadata.append(dict(frame=t,time_s=t/30,graphs=records))
    # Compare objective effects at the SAME baseline state; report local arm gradient, not causal percentages.
    terms=('interaction_mesh','fingertip_tracking','source_contact','finger_bones','palm_alignment','omni_arm_prior','finger_init_prior')
    samples={term:[] for term in (*terms,'temporal')}
    for t in range(len(q)):
        if not contact[t].any():continue
        target=ref.frame_target(human[t],O,pose[t],contact[t],surface[t])
        residual,J=ref.residuals(q[t],target,Qbody[t],init[t],q[t-1] if t else None)
        offset=0;blocks={term:[] for term in samples}
        for side in range(2):
            sizes=[3*(21+len(O)),15,3*int(contact[t,side].sum()),45,6,7,20]
            for term,count in zip(terms,sizes):
                ids=np.arange(offset,offset+count);offset+=count;blocks[term].extend(ids)
        if t:blocks['temporal'].extend(range(offset,offset+54));offset+=54
        assert offset==len(residual)
        cols=np.concatenate([ref.arm_columns[s] for s in range(2) if contact[t,s].any()])
        for term,ids in blocks.items():
            rr=residual[ids];jj=J[ids][:,cols]
            samples[term].append([float(rr@rr),float(np.linalg.norm(2*jj.T@rr))])
    influence={k:dict(mean_weighted_cost=float(np.mean(v,axis=0)[0]),mean_contact_arm_gradient_norm=float(np.mean(v,axis=0)[1])) for k,v in samples.items()}
    write(target_dir/'graphs.json',dict(sequence=seq,representative_frame=chosen,frames=metadata,
        body_landmarks=names,hand_landmarks='palm/wrist, then thumb/index/middle/ring/little MCP-PIP-DIP-TIP',
        stored_graph_coordinates='object frame',display_coordinates='world frame',
        source_paths={k:str(v) for k,v in info.items()},
        body_graph_weight=10,body_neighbours='uniform 1/degree',coupled_graph_weight=500,
        coupled_normalization='divided by 21+N_object, fingertip row weight 10',
        coupled_neighbours='exp(-30*source_edge_length)/(sum of neighbour exponentials + 1e-9)',
        measured_at='Saved current V2 state; frames with any source contact; arm gradients restricted to contacting sides.',
        influence=influence,warning='Gradient norms are local numerical sensitivity, not fractions of causal influence. Constraints, competing gradients, warm starts and nonlinear convergence also affect the answer. Body/coupled mesh costs are not comparable across different nodes/weights.'))
    fig,ax=plt.subplots(figsize=(9,4.7));keys=list(influence);values=[influence[k]['mean_contact_arm_gradient_norm'] for k in keys]
    ax.barh(keys,values,color=['#377ca7' if k!='omni_arm_prior' else '#e38834' for k in keys]);ax.set_xscale('symlog',linthresh=1e-4)
    ax.set_xlabel('Mean norm of weighted objective gradient on contacting arm joints')
    ax.set_title('Local influence at current V2 poses (not causal attribution)');fig.tight_layout();fig.savefig(target_dir/'arm_gradient.png',dpi=150);plt.close(fig)
    print('GRAPHS',seq,flush=True)


if __name__=='__main__':
    for seq in sys.argv[1:] or COHORT:main(seq)
