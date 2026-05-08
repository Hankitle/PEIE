from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .pyiqa import pyiqa_niqe, pyiqa_brisque, pyiqa_nima

__all__ = ['pyiqa_niqe', 'pyiqa_brisque', 'pyiqa_nima']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
