"""CPU software-rendered matched v1/v2 videos and contact-frame PNGs."""
from pathlib import Path
import json
import os
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = '/usr/share/glvnd/egl_vendor.d/50_mesa.json'
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
os.environ['LP_NUM_THREADS'] = '1'
os.environ.pop('IMAGEIO_FFMPEG_EXE', None)  # imageio's bundled build includes libx264
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

WORK = Path(__file__).resolve().parents[4]
ROOT = WORK.parent
BATCH = ROOT/'holosoma/eval_out/ref_candidates/batch_20261011_current_120'
OUT = ROOT/'holosoma/eval_out/ref_candidates/pipeline_v2_pilot_20261018'


def render(seq):
    start = time.perf_counter()
    job = json.loads((BATCH/'jobs'/f'{seq}.json').read_text())
    clip = OUT/'clips'/seq
    q = [np.load(job['final_npz'])['qpos'], np.load(clip/'final_v2.npz')['qpos']]
    assert q[0].shape == q[1].shape
    m = mujoco.MjModel.from_xml_path(job['model_xml']);m.vis.global_.offwidth=480;m.vis.global_.offheight=480
    d = mujoco.MjData(m);renderer = mujoco.Renderer(m, 480, 480)
    cam = mujoco.MjvCamera();cam.azimuth=210;cam.elevation=-20
    visible = np.flatnonzero((m.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)&(m.geom_rgba[:,3]>0))
    centers, radii = [], []
    for t in range(len(q[0])):
        center = .5*(q[0][t,:3]+q[0][t,-7:-4]);center[2] = .8
        centers.append(center)
        for trajectory in q:
            d.qpos[:]=trajectory[t];mujoco.mj_kinematics(m,d)
            radii.append(np.max(np.linalg.norm(d.geom_xpos[visible]-center,axis=1)+m.geom_rbound[visible]))
    cam.distance=max(2.5,float(max(radii))*1.06/np.sin(np.deg2rad(m.vis.global_.fovy/2)))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    def frame(t, close=False):
        canvas=Image.new('RGB',(960,520),(245,247,250));draw=ImageDraw.Draw(canvas)
        cam.lookat[:]=centers[t]
        olddistance=cam.distance
        if close:
            z=np.load(BATCH/'clips'/seq/'part0'/f'{seq}_canonical.npz')
            cam.lookat[:]=z['hand_kp_scaled'][t,:,[4,8,12,16,20],:].reshape(-1,3).mean(0)
            cam.distance=1.1
        for i,trajectory in enumerate(q):
            d.qpos[:]=trajectory[t];mujoco.mj_forward(m,d)
            renderer.update_scene(d,cam);canvas.paste(Image.fromarray(renderer.render()),(480*i,40))
            draw.text((480*i+12,10),('V1: existing reference','V2: joint arms + fingers')[i],fill=(20,30,45),font=font)
        cam.distance=olddistance
        draw.text((12,495),f'{t/30:.2f}s / {(len(q[0])-1)/30:.2f}s',fill=(0,0,0),font=font)
        return np.asarray(canvas)
    rend=OUT/'renders';rend.mkdir(exist_ok=True)
    with imageio.get_writer(rend/f'{seq}.mp4',fps=15,codec='libx264',quality=8,macro_block_size=1,
                           ffmpeg_params=['-threads','1','-movflags','+faststart']) as writer:
        for t in range(0,len(q[0]),2):
            writer.append_data(frame(t))
    with np.load(BATCH/'evaluation_v1/per_frame'/f'{seq}.npz') as raw:
        active=raw['source_contact'].sum(axis=(1,2))
        ids=np.flatnonzero(active==active.max());t=int(ids[len(ids)//2])
    imageio.imwrite(rend/f'{seq}.png',frame(t))
    imageio.imwrite(rend/f'{seq}_hands.png',frame(t,True))
    renderer.close()
    result=dict(sequence=seq,frames_source=len(q[0]),video_frames=len(range(0,len(q[0]),2)),video_fps=15,
                source_fps=30,contact_png_frame=t,render_seconds=time.perf_counter()-start)
    (rend/f'{seq}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    render(sys.argv[1])
