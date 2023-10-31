import os
import glob
import tqdm
from PIL import Image
import numpy as np


def map_indices(x, ids, dest=0):
    if len(ids):
        x[np.isin(x, ids)] = dest
    return x

def invmap_indices(x, ids, dest=0):
    if len(ids):
        x[~np.isin(x, ids)] = dest
    return x


def main(work_dir, ids, confirm=False, info=False):
    os.makedirs(f'{work_dir}/masks_preview', exist_ok=True)
    ids = [int(i) for i in ids]
    pbar = tqdm.tqdm(glob.glob(f'{work_dir}/masks/*.png'))
    for f in pbar:
        im = Image.open(f)
        x = np.array(im)
        if info:
            pbar.set_description(f'{np.unique(x)}')
            continue

        x2 = x.copy()
        x2 = invmap_indices(x2, ids, 0)

        if confirm:
            f2 = f'{work_dir}/masks/{os.path.basename(f)}'
            im2 = Image.fromarray(x2).convert('P')
            im2.putpalette(im.palette)
            im2.save(f2)
        else:
            f2 = f'{work_dir}/masks_preview/{os.path.basename(f)}'
            im2 = Image.fromarray(np.concatenate([x, x2])).convert('P')
            im2.putpalette(im.palette)
            im2.save(f2)


if __name__ == '__main__':
    import fire
    fire.Fire(main)