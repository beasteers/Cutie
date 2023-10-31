import os
import glob
import tqdm
from PIL import Image
import numpy as np
import supervision as sv

def main(work_dir):
    work_dir = work_dir.rstrip('/')
    os.makedirs(f'{work_dir}/masks_preview', exist_ok=True)
    fs = sorted(glob.glob(f'{work_dir}/masks/*.png'))
    h, w = np.array(Image.open(fs[0])).shape
    with sv.VideoSink(f'{work_dir}/masks.mp4', sv.VideoInfo(width=w, height=h*2, fps=30)) as s:
        for f in tqdm.tqdm(fs):
            fim = f'{work_dir}/images/{os.path.splitext(os.path.basename(f))[0]}.jpg'
            if not os.path.isfile(fim):
                print('missing', fim)
                continue
            im = np.array(Image.open(fim))[:,:,::-1]
            mask_im = Image.open(f)
            mask = np.array(mask_im)
            immask = np.array(mask_im.convert('RGB'))[:,:,::-1]
            # pal = np.array(mask_im.palette.getdata())
            # mask = pal[mask]
            im[mask != 0] = im[mask != 0] / 2 + immask[mask != 0] / 2
            s.write_frame(np.concatenate([im, immask]))



if __name__ == '__main__':
    import fire
    fire.Fire(main)