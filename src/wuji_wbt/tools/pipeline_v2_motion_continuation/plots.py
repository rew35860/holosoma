"""Supplementary full-clip and event-window plots from saved evaluated outputs."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import *
from solver import load

def main():
    records=[]
    for seq in COHORT:
        _,_,_,ref,_,_=load(seq,'baseline');fingers=np.concatenate(ref.finger_columns)
        fig,axes=plt.subplots(4,2,figsize=(15,12),constrained_layout=True)
        bounds=(3.45,4.05) if 'organizer' in seq else (2.4,3.05)
        for variant in VARIANTS:
            dest=destination(seq,variant);Q=np.load(dest/'final_v2.npz')['qpos'][:,ref.qadr]
            raw=np.load(dest/'contact_geometry_per_frame.npz');times=np.arange(len(Q))/30
            depth=raw['max_depth_object_ground_self_m'][:,0]*1000
            self_depth=raw['max_depth_object_ground_self_m'][:,2]*1000
            change=np.r_[0.,np.degrees(np.max(np.abs(np.diff(Q[:,fingers],axis=0)),axis=1))]
            acceleration=np.r_[0.,0.,np.max(np.abs(np.diff(Q[:,fingers],n=2,axis=0)*900),axis=1)]
            for col in range(2):
                axes[0,col].plot(times,depth,label=LABELS[variant])
                axes[1,col].plot(times,self_depth,label=LABELS[variant])
                axes[2,col].plot(times,change,label=LABELS[variant])
                axes[3,col].plot(times,acceleration,label=LABELS[variant])
        for col in range(2):
            axes[0,col].set_ylabel('Max object penetration (mm)')
            axes[1,col].set_ylabel('Max self-penetration (mm)')
            axes[2,col].set_ylabel('Largest finger change (degrees/frame)')
            axes[3,col].set_ylabel('Largest finger acceleration (rad/s²)')
            axes[3,col].set_xlabel('Source time (s)')
            axes[0,col].axhline(1,color='gray',ls=':',lw=1)
            axes[2,col].axhline(np.degrees(.2),color='gray',ls=':',lw=1)
            for row in range(4):
                axes[row,col].grid(alpha=.2)
                axes[row,col].set_xlim(bounds if col else (0,times[-1]))
        axes[0,0].set_title('Full clip (includes any shifted glitches)')
        axes[0,1].set_title('Previously identified glitch window')
        axes[0,0].legend(fontsize=9)
        fig.suptitle(seq+'\nGeometry side; signed recovery OFF; thumb/index x1; original body/object fixed')
        folder=OUT/'plots';folder.mkdir(exist_ok=True)
        fig.savefig(folder/f'{seq}.png',dpi=150);plt.close(fig)
        records.append(str(folder/f'{seq}.png'))
    write(OUT/'plot_manifest.json',dict(script_sha256=sha(__file__),plots=records,source='Saved evaluated trajectories; no optimization or metric-definition changes.'))

if __name__=='__main__':main()
