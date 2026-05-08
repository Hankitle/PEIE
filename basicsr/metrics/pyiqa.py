from basicsr.utils import img2tensor
from basicsr.utils.registry import METRIC_REGISTRY

import pyiqa

def register_metric(metric_name, device='cuda'):
    """
    A helper function to register metrics with lazy initialization.
    """
    if not hasattr(register_metric, metric_name):
        setattr(register_metric, metric_name, pyiqa.create_metric(metric_name, device=device))
    return getattr(register_metric, metric_name)

@METRIC_REGISTRY.register()
def pyiqa_niqe(img, path=None, **kwargs):
    niqe = register_metric('niqe')
    if path is not None:
        return niqe(path).item()
    return niqe(img2tensor(img / 255.0).unsqueeze(0)).item()

@METRIC_REGISTRY.register()
def pyiqa_brisque(img, path=None, **kwargs):
    brisque = register_metric('brisque')
    if path is not None:
        return brisque(path).item()
    return brisque(img2tensor(img / 255.0).unsqueeze(0)).item()

@METRIC_REGISTRY.register()
def pyiqa_nima(img, path=None, **kwargs):
    nima = register_metric('nima')
    if path is not None:
        return nima(path).item()
    return nima(img2tensor(img / 255.0).unsqueeze(0)).item()

@METRIC_REGISTRY.register()
def pyiqa_musiq(img, path=None, **kwargs):
    musiq = register_metric('musiq')
    if path is not None:
        return musiq(path).item()
    return musiq(img2tensor(img / 255.0).unsqueeze(0)).item()

@METRIC_REGISTRY.register()
def pyiqa_clipiqa(img, path=None, **kwargs):
    clipiqa = register_metric('clipiqa')
    if path is not None:
        return clipiqa(path).item()
    return clipiqa(img2tensor(img / 255.0).unsqueeze(0)).item()

@METRIC_REGISTRY.register()
def pyiqa_paq2piq(img, path=None, **kwargs):
    paq2piq = register_metric('paq2piq')
    if path is not None:
        return paq2piq(path).item()
    return paq2piq(img2tensor(img / 255.0).unsqueeze(0)).item()
