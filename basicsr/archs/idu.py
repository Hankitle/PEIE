import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidPool(nn.Module):
    '''Pyramid Pooling

    Args:
        num_feat (int): Number of channels.
        pool_sizes (list): Pooling sizes.
        pool_type (str): Pooling type.
    '''
    def __init__(self, num_feat, pool_sizes, pool_type):
        super(PyramidPool, self).__init__()

        self.pool_sizes = pool_sizes
        self.pool = self._get_pooling_function(pool_type)

        convs = []
        # a list of convolutional layers for each pool size
        for _ in range(len(pool_sizes)):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(num_feat, 1, 1, 1, 0),
                    nn.ReLU(inplace=True)
                ))
        self.convs = nn.ModuleList(convs)

    def _get_pooling_function(self, pool_type):
        if pool_type == 'avg':
            return F.avg_pool2d
        elif pool_type == 'max':
            return F.max_pool2d
        elif pool_type == 'min':
            return lambda x, *args, **kwargs: -F.max_pool2d(-x, *args, **kwargs)
        elif pool_type == 'adp_avg':
            return F.adaptive_avg_pool2d
        elif pool_type == 'adp_max':
            return F.adaptive_max_pool2d
        elif pool_type == 'adp_min':
            return lambda x, *args, **kwargs: -F.adaptive_max_pool2d(-x, *args, **kwargs)
        else:
            raise ValueError('Unknown pool type')

    def forward(self, x):
        pool_slices = []
        h, w = x.shape[2:]

        for module, pool_size in zip(self.convs, self.pool_sizes): 
            out = self.pool(x, pool_size)
            out = module(out)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
            pool_slices.append(out)

        return pool_slices

class IDU(nn.Module):
    """ IDU: Illumination-Adaptive Dehazing Unit

    Args:
        num_feat (int): Number of channels.
        depth (int): Depth of the IDU.
    """
    def __init__(self, num_feat, depth):
        super(IDU, self).__init__()
        scale = 2**depth if depth < 2 else 4
        pool_list = [4//scale, 8//scale, 12//scale, 16//scale, 24//scale]
        self.feat = PyramidPool(num_feat, pool_list, 'avg')
        self.le = nn.Sequential(
                    nn.Conv2d(num_feat+len(pool_list)-1, num_feat//8, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_feat//8, num_feat, 3, 1, 1),
                    nn.Sigmoid()
                )
        self.li = nn.Sequential(
                    nn.Conv2d(num_feat+len(pool_list)-1, num_feat//8, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_feat//8, num_feat, 3, 1, 1),
                    nn.Sigmoid()
                )
        self.t = nn.Sequential(
                    nn.Conv2d(num_feat+len(pool_list)-1, num_feat//8, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_feat//8, num_feat, 3, 1, 1),
                    nn.Sigmoid()
                )
        

    def forward(self, x):
        pool_slices = self.feat(x)
        fine_feat = torch.cat([x] + pool_slices[:-1], dim=1)
        coarse_feat = torch.cat([x] + pool_slices[1:], dim=1)
        le = self.le(coarse_feat)
        li = self.li(fine_feat)
        t = self.t(fine_feat)
        li_t = li * t
        r = x * li_t + le * (li - li_t)
        return r