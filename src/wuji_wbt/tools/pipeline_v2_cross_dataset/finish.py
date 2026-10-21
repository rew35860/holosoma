"""Apply the frozen selection rule, complete missing validation, render examples."""
import os,time,subprocess
from config import *
import worker

def main():
    # Input-repair helpers can outlive the original batch outcome record.
    while any(subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True).stdout.strip()=='active'
              for unit in ('v2-cross-clothrack-repair-20261021.service','v2-cross-organizer-recovery-20261021.service')):
        time.sleep(15)
    selected=worker.choose()
    if selected!='current':
        missing=[(r,selected) for r in rows() if r['split']=='validation' and not
            (OUT/'variants'/selected/'clips'/r['sequence']/'evaluate_completed.json').exists()]
        if missing:worker.phase('validation',missing)
    subprocess.run([PY,str(CODE/'validate.py')],check=True)
    subprocess.run([PY,str(CODE/'report.py')],check=True)
    import recommendation
    selected=recommendation.main()
    records=read(OUT/'per_sequence.json');examples=[]
    for dataset in ('humoto','hiphi'):
        eligible=[r for r in records if r['dataset']==dataset and r['variant']==selected and r['split']=='validation']
        # Show a favorable and middle-ranked result, not only the best clip.
        eligible.sort(key=lambda r:r['patch_gap_mm'])
        if eligible:
            for i in sorted(set([0,len(eligible)//2])):examples.append(eligible[i]['sequence'])
    write(OUT/'render_selection.json',dict(sequences=examples,variant=selected,
        rule='Lowest and median intended-patch gap among successful validation clips per dataset; all clip metrics and failures remain reported. This is a visualization sample, not the validation denominator.'))
    outcomes=[]
    for seq in examples:
        log=OUT/'logs'/seq/'render.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as stream:
            r=subprocess.run([PY,str(CODE/'render.py'),seq],env=dict(os.environ,CROSS_VARIANT=selected),stdout=stream,stderr=subprocess.STDOUT)
        outcomes.append(dict(sequence=seq,returncode=r.returncode))
        write(OUT/'status.json',dict(state='rendering',selected=selected,rendered=len(outcomes),total=len(examples),outcomes=outcomes))
    subprocess.run([PY,str(CODE/'report.py')],check=True)
    write(OUT/'status.json',dict(state='complete' if all(r['returncode']==0 for r in outcomes) else 'render_failure',
        selected=selected,sequences=len(rows()),render_outcomes=outcomes,
        note='Pilot attempts complete; motion failures remain in failures.json and the report. Complete does not mean every motion succeeded.'))
if __name__=='__main__':main()
