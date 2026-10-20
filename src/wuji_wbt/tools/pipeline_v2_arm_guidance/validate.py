"""Validate completed experiment artifacts without rerunning optimization."""
import subprocess
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import numpy as np
import imageio_ffmpeg
from config import *
import diagnostics


def main():
    manifest=read(OUT/'input_manifest.json')
    assert sha(SOLVER)==manifest['solver_sha256']
    assert sha(PILOT_CODE.parent/'ref_candidates_metrics/evaluate.py')==manifest['evaluator_sha256']
    successes=[];media=[]
    for variant in REMOVALS:
        conf=read(OUT/'settings'/f'{variant}.json')
        changed={k:v for k,v in conf.items() if v!=BASE_SETTINGS[k]}
        assert changed==REMOVALS[variant]
        for seq in COHORT:
            dest=OUT/'variants'/variant/'clips'/seq
            if not (dest/'completed.json').exists():continue
            check=read(dest/'checks.json')
            assert check['passed'] and check['remaining_body_exact'] and check['source_contact_mask_exact']
            assert check['initialization_exact'] and check['settings_exact'] and check['joint_limits_passed']
            if variant=='full':assert check['full_v2_trajectory_max_difference']<1e-7
            assert read(dest/'run.json')['settings']==conf
            assert sha(dest/'final_v2.npz')==read(dest/'completed.json')['final_sha256']
            if not (dest/'comfort_per_joint.json').exists():diagnostics.main(variant,seq)
            assert (dest/'collision_audit.json').exists()
            successes.append([variant,seq])
    for seq in COHORT:
        graph=read(OUT/'graphs'/seq/'graphs.json')
        assert graph['frames'][0]['graphs'][0]['human_robot_nodes']==15
        assert all(g['human_robot_nodes']==21 for g in graph['frames'][0]['graphs'][1:])
        for p,expected in manifest['input_sha256'][seq].items():assert sha(p)==expected
        if not (OUT/'renders'/f'{seq}_posture.png').exists():
            subprocess.run([PY,str(CODE/'render.py'),seq,'--preview'],cwd=CODE,check=True)
        metadata=read(OUT/'renders'/f'{seq}.json')
        reader=imageio_ffmpeg.read_frames(str(OUT/'renders'/f'{seq}.mp4'),
            input_params=['-threads','1'],output_params=['-threads','1'])
        header=next(reader);count=sum(1 for frame in reader)
        stream=dict(width=header['size'][0],height=header['size'][1],
                    r_frame_rate=f"{header['fps']:g}/1",nb_read_frames=count)
        assert int(stream['nb_read_frames'])==metadata['video_frames']
        assert stream['r_frame_rate']=='15/1' and stream['height']==540
        assert stream['width']==480*len(metadata['panels'])
        media.append(dict(sequence=seq,**stream))
    subprocess.run([PY,str(CODE/'report.py')],cwd=CODE,check=True)
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key not in ('src','href','poster') or not value:continue
                parsed=urlsplit(value)
                if parsed.scheme or value.startswith('#'):continue
                assert (OUT/unquote(parsed.path)).exists(),value
    Links().feed((OUT/'index.html').read_text())
    status=read(OUT/'status.json')
    assert not status['pending'] and status['state'] in ('complete','failed')
    assert len(successes)+len(status['errors'])==status['total']==18
    assert sum(v=='full' for v,s in successes)==len(COHORT)
    write(OUT/'validation.json',dict(passed=True,completed_trials=len(successes),planned_trials=18,
        successful_trials=successes,solver_unchanged=True,frozen_evaluator_unchanged=True,
        input_hashes_unchanged=True,only_contact_arm_prior_varies=True,
        remaining_body_and_initialization_exact=True,all_six_baselines_reproduce=True,
        media=media,all_local_report_links_resolve=True,
        experiment_script_sha256={p.name:sha(p) for p in CODE.glob('*.py')},
        failed_trials=read(OUT/'status.json')['errors']))


if __name__=='__main__':main()
