import torch


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


def get_config(cfg=None, config_name='inference_config', version_base='1.3.2', config_path="config", job_name="demo"):
    from omegaconf import DictConfig
    if not isinstance(cfg, DictConfig):
        with torch.inference_mode():
            from hydra import compose, initialize
            initialize(version_base=version_base, config_path=config_path, job_name=job_name)
            update = cfg
            cfg = compose(config_name=config_name)
            if update:
                cfg.update(update)
    return cfg
