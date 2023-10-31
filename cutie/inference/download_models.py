import os
import hashlib


_links = {
    'coco_lvis_h18_itermask': ('https://github.com/hkchengrex/Cutie/releases/download/v1.0/coco_lvis_h18_itermask.pth', '6fb97de7ea32f4856f2e63d146a09f31'),
    'cutie-base-mega': ('https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth', 'a6071de6136982e396851903ab4c083a'),
    # TODO add md5
    'cutie-base-nomose': ('https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-nomose.pth', ''),
    'cutie-base-wmose': ('https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-wmose.pth', ''),
    'cutie-small-mega': ('https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-small-mega.pth', ''),
    'cutie-small-nomose': ('https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-small-nomose.pth', ''),
    'cutie-small-wmose': ('https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-small-wmose.pth', ''),
}

def download_from_link(link, md5, output_dir='weights', block_size=1024):
    os.makedirs('output', exist_ok=True)
    # download file if not exists with a progressbar
    filename = link.split('/')[-1]
    path = os.path.join(output_dir, filename)
    if not os.path.exists(path) or (md5 and hashlib.md5(open(path, 'rb').read()).hexdigest() != md5):
        import requests
        from tqdm import tqdm

        print(f'Downloading {filename}...')
        r = requests.get(link, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        with open(path, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True) as t:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        if total_size != 0 and t.n != total_size:
            raise RuntimeError('Error while downloading %s' % filename)
    return path

def download(model='cutie-base-mega', md5=None):
    if model in _links:
        model, md5 = _links[model]
    if model.startswith('https://') or model.startswith('http://'):
        pass
    else:
        possible = os.path.splitext(os.path.basename(model))[0]
        if possible in _links:
            model, md5 = _links[possible]
        else:
            raise KeyError(f'{model} not in {_links.keys()}')
    return download_from_link(model, md5)

def download_models_if_needed(*models):
    for link, md5 in models or ['rltm', 'cutie-base-mega']:
        download(link, md5)

if __name__ == '__main__':
    download_models_if_needed()