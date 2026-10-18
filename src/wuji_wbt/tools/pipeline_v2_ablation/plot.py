"""Export the complete study as a standalone figure; dots are clips, not error bars."""
import os
os.environ['MPLBACKEND']='Agg'
import numpy as np
import matplotlib.pyplot as plt
from config import *

def main():
    summary=read(OUT/'summary.json')
    assert summary['state']!='running'
    cohort=summary['shared_successful_sequences']
    assert cohort
    variants=list(REMOVALS)
    names={'full':'Full V2','no_finger_prior':'No finger\nprior','no_contact':'No contact\nattraction',
           'no_palm':'No palm\nalignment','no_fingertip':'No fingertip\ntracking',
           'no_arm_prior':'No arm\nprior','core_only':'None of\nthe five',
           'no_priors':'No finger +\narm priors'}
    fields=[('contact_f1_pct','Contact F1 (%) ↑'),
            ('combined_pen_duration_pct','Penetration duration (%) ↓'),
            ('primary_tip_error_mm','Thumb/index error (mm) ↓'),
            ('palm_normal_error_deg','Palm error (degrees) ↓'),
            ('body_bone_direction_error_deg','Body direction error (degrees) ↓'),
            ('measured_refinement_process_seconds','Refinement process time (s) ↓')]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained')
    colors=['#276c9b' if v=='full' else '#ce7430' if v=='core_only' else '#64a59a' for v in variants]
    for ax,(key,label) in zip(axes.flat,fields):
        vals=[]
        for v in variants:
            if key.startswith('measured_'):
                vals.append([read(OUT/'variants'/v/'clips'/seq/'completed.json')[key] for seq in cohort])
            else:
                vals.append([read(OUT/'variants'/v/'clips'/seq/'metrics.json')['v2'][key] for seq in cohort])
        means=np.mean(vals,axis=1);x=np.arange(len(variants))
        ax.bar(x,means,color=colors,alpha=.82,width=.7)
        for i,v in enumerate(vals):
            ax.scatter(i+np.linspace(-.2,.2,len(v)),v,s=22,facecolors='white',edgecolors='#273444',zorder=3,linewidth=.6)
        ax.set_xticks(x,[names.get(v,LABELS[v]) for v in variants],rotation=35,ha='right')
        ax.set_title(label,loc='left',fontweight='bold');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    fig.suptitle(f'Wuji V2 objective ablation — {len(cohort)} jointly completed HUMOTO clips\nBars: equal-weight clip means; dots: clips. Failed runs are reported separately; this is the shared successful subset.',fontsize=15)
    fig.supxlabel('Contact threshold: 30 mm. Penetration: object >20 mm or ground/self >10 mm. Runtime excludes the reused body stage.',fontsize=10)
    fig.savefig(OUT/'ablation_summary.png',dpi=180)
    fig.savefig(OUT/'ablation_summary.pdf')
    plt.close(fig)

if __name__=='__main__':main()
