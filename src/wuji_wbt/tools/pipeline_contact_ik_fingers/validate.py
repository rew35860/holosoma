"""Verify the completed pilot's invariants, numerical checks, and media."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import subprocess
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import numpy as np
import mujoco
import imageio_ffmpeg
from PIL import Image
from config import *
from solver import ArmIK,SETTINGS


def main():
    protocol=read(OUT/'protocol.json')
    for path,digest in protocol['input_sha256'].items():assert sha(path)==digest,path
    for path,digest in protocol['source_sha256'].items():assert sha(path)==digest,path
    info=inputs();m=mujoco.MjModel.from_xml_path(str(info['model']));ref=ArmIK(m,info['object'])
    base=np.load(info['baseline'])['qpos'];dest=OUT/'variants/contact_ik/clips'/SEQ
    q=np.load(dest/'final_v2.npz')['qpos'];frozen=np.setdiff1d(np.arange(m.nq),ref.qadr)
    assert q.shape==base.shape and np.isfinite(q).all()
    assert np.array_equal(q[:,frozen],base[:,frozen])
    assert np.max(np.abs(q[:,ref.qadr[ref.fingers]]-base[:,ref.qadr[ref.fingers]]))>1e-3
    assert np.all(q[:,ref.qadr]>=ref.lo-1e-6) and np.all(q[:,ref.qadr]<=ref.hi+1e-6)
    maxrate=float(np.abs(np.diff(q[:,ref.qadr],axis=0)).max())
    assert maxrate<=SETTINGS['rate_limit']+1e-6
    raw=np.load(dest/'per_frame_metrics.npz');source=np.load(info['mask'])['source_contact']
    assert np.array_equal(raw['source_contact'],source)
    numeric=read(OUT/'numerical_checks.json');assert numeric['passed']
    run=read(dest/'run.json');diag=read(dest/'ik_diagnostics.json')
    assert run['feasible_frames']==diag['collision_feasible_frames']
    physical=read(OUT/'physical_contact_audit.json')
    assert all(not p['zero_gradient_but_nonrigid_pairs'] for p in physical.values())
    images=list((OUT/'renders').glob('*.png'))
    for path in images:
        with Image.open(path) as im:im.verify()
    video=OUT/'renders'/f'{SEQ}.mp4'
    os.environ.pop('IMAGEIO_FFMPEG_EXE',None)
    decoder=imageio_ffmpeg.read_frames(str(video),input_params=['-threads','1'],output_params=['-threads','1'])
    media=next(decoder);decoded=0
    for frame in decoder:
        assert len(frame)==1920*540*3
        decoded+=1
    assert decoded==len(range(0,len(q),2)) and tuple(media['size'])==(1920,540)
    assert abs(media['fps']-15)<1e-9
    media['decoded_frames']=decoded
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key not in ('href','src','poster'):continue
                parsed=urlsplit(value)
                if parsed.scheme or not parsed.path:continue
                assert (OUT/unquote(parsed.path)).exists(),value
    Links().feed((OUT/'index.html').read_text())
    validation=dict(passed=True,frames=len(q),active_dofs=54,arm_dofs=14,finger_dofs=40,frozen_body_object_exact=True,
        source_contact_mask_exact=True,joint_limits_passed=True,maximum_active_change_rad_per_frame=maxrate,
        scientific_inputs_and_frozen_evaluator_unchanged=True,geometry_derivative_checks_passed=True,
        zero_gradient_exceptions_have_identical_active_joint_chains=True,
        solver_feasible_frames=run['feasible_frames'],frozen_geometry_violation_frames=run['frozen_geometry_violation_frames'],
        pngs_verified=len(images),video_full_decode_passed=True,video=media,local_html_links_passed=True,
        final_trajectory_sha256=sha(dest/'final_v2.npz'),video_sha256=sha(video),
        caution='Implementation/media validation is not a claim that every frame is collision-free or that quality improved.')
    write(OUT/'validation.json',validation)
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for path in CODE.glob('*.py'):(snapshot/path.name).write_bytes(path.read_bytes())
    status=read(OUT/'status.json')
    page=(OUT/'index.html').read_text()
    marker='<section id="validation">'
    if marker in page:page=page.split(marker)[0]
    page+=marker+'<h2>Validation and time</h2><p><a href="validation.json">Verified fixed coordinates, source masks, joint limits, input hashes, geometry Jacobians and complete MP4 decoding</a>.</p>'
    page+='<p>Measured sequential process times: '+', '.join(f'{k}: {v:.1f} s' for k,v in status['stages'].items())+f'. Total: {status["elapsed_seconds"]:.1f} s. Up to three low-priority CPU workers for evaluation; one sequential solve worker; no GPU allocation.</p>'
    failures=[r for r in read(dest/'iterations.json') if not r['collision_feasible']]
    write(OUT/'unresolved_frames.json',failures)
    page+=f'<details><summary>{len(failures)} frames where this local solve did not satisfy every controllable clearance requirement</summary>'
    page+='<p>These are solver outcomes, not proofs that the corresponding contact targets are unreachable. The video includes every frame, including these diagnostic candidates. Frame numbers are zero-based.</p><table><tr><th>Frame</th><th>Time (s)</th><th>Largest clearance shortfall (mm)</th></tr>'
    for r in failures:page+=f'<tr><td>{r["frame"]}</td><td>{r["frame"]/30:.3f}</td><td>{r["controllable_violation_mm"]:.4f}</td></tr>'
    page+='</table><p><a href="unresolved_frames.json">Full per-frame solver diagnostics</a></p></details></section>'
    (OUT/'index.html').write_text(page)
    print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
