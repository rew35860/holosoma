"""Extract human/selected-geometry views from existing renders, without rerendering."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
import numpy as np
import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
from config import OUT, COHORT


def run(seq):
    dest = OUT / 'diagnostics/jitter_3s' / seq
    start, end = (3.3, 4.1) if 'organizer' in seq else (2.35, 3.05)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    for suffix in ('', '_hands'):
        source = OUT/'renders'/f'{seq}{suffix}.mp4'
        reader = imageio_ffmpeg.read_frames(str(source), input_params=['-threads', '1'], output_params=['-threads', '1'])
        metadata = next(reader); width, height = metadata['size']; fps = metadata['fps']
        chosen = []
        for i, raw in enumerate(reader):
            t = i / fps
            if t < start or t > end: continue
            a = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
            frame = Image.fromarray(np.concatenate([a[:, :480], a[:, 960:1440]], axis=1))
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 960, 39), fill=(245, 247, 250))
            draw.text((10, 10), 'Human demonstration', font=font, fill=(20, 30, 45))
            draw.text((490, 10), 'Geometry side; recovery OFF; x1', font=font, fill=(20, 30, 45))
            draw.rectangle((0, 517, 960, 540), fill=(245, 247, 250))
            draw.text((10, 519), f'Source {t:.3f} s | frame {round(t*30)} | playback 1/3 speed', font=font, fill=(20, 30, 45))
            chosen.append(frame)
        assert chosen
        with imageio.get_writer(dest/f'event{suffix}_slow.mp4', fps=fps/3, codec='libx264', quality=8,
                                macro_block_size=1, ffmpeg_params=['-threads', '1', '-movflags', '+faststart']) as writer:
            for frame in chosen: writer.append_data(np.asarray(frame))
        indices = np.linspace(0, len(chosen)-1, 4).round().astype(int)
        sheet = Image.new('RGB', (960, 540*4), 'white')
        for row, i in enumerate(indices): sheet.paste(chosen[i], (0, row*540))
        sheet.save(dest/f'event{suffix}_frames.png')
        print(dest/f'event{suffix}_slow.mp4', flush=True)


if __name__ == '__main__':
    for seq in COHORT: run(seq)
