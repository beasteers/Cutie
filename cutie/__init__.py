import torch
from omegaconf import DictConfig, open_dict


def get_model(cfg=None, model=None):
    from .inference import InferenceCore

    cfg = get_config(cfg)
    if model is None:
        model = get_base_model(cfg)
    return InferenceCore(model, cfg)


def get_base_model(cfg=None):
    from .model.cutie import CUTIE
    from .inference import download_models
    cfg = get_config(cfg)
    download_models.download(cfg.weights)
    model = CUTIE(cfg).eval()
    model.load_weights(torch.load(cfg.weights))
    return model


def get_config(cfg=None):
    if not isinstance(cfg, DictConfig):
        with torch.inference_mode():
            from hydra import compose, initialize
            initialize(version_base='1.3.2', config_path="config", job_name="demo")
            data = cfg
            cfg = compose(config_name="inference_config")
            if cfg:
                cfg.update(data)
    return cfg
