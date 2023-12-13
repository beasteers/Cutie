dependencies = ['torch', 'scipy', 'hydra-core', 'requests', 'tqdm']


def cutie(config={}):
    """Cutie tracker."""
    import cutie
    return cutie.get_model(config)


def base_model(config={}):
    """Cutie base model."""
    import cutie
    return cutie.get_base_model(config)
