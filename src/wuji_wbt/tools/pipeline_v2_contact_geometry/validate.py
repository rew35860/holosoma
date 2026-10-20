"""Check inputs unchanged, all trajectories evaluated, and decode both videos."""
import numpy as np
from PIL import Image
import imageio_ffmpeg
from config import *

def main():
    protocol=read(OUT/'protocol.json');checks=[]
    for group in ('input_sha256','source_sha256','target_sha256'):
        for path,expected in protocol[group].items():assert sha(path)==expected,path
    assert read(OUT/'preflight.json')['passed']
    for seq in COHORT:
        for variant in REMOVALS:
            dest=destination(seq,variant);Q=np.load(dest/'final_v2.npz')['qpos']
            assert np.isfinite(Q).all()
            assert read(dest/'diagnostics.json')['frozen_body_object_exact']
            assert read(dest/'metrics.json')['unchanged_source_contact_mask']
            assert read(dest/'metrics.json')['unchanged_body_exact']
        for suffix in ('','_hands','_posture'):
            with Image.open(OUT/'renders'/f'{seq}{suffix}.png') as im:im.verify()
        path=OUT/'renders'/f'{seq}.mp4'
        decoder=imageio_ffmpeg.read_frames(str(path),input_params=['-threads','1'],output_params=['-threads','1'])
        meta=next(decoder);count=sum(1 for _ in decoder)
        expected=(len(Q)+1)//2
        assert count==expected,(seq,count,expected)
        assert meta['size']==(2400,540),meta
        checks.append(dict(sequence=seq,video_frames=count,size=meta['size'],video_sha256=sha(path)))
    write(OUT/'validation.json',dict(passed=True,inputs_and_sources_unchanged=True,videos=checks))

if __name__=='__main__':main()
