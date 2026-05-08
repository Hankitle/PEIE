import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY
from .idu import IDU
 
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    
@ARCH_REGISTRY.register()
class IADNet(nn.Module):
    """ IADNet network architecture.

    Args:
        num_feat (int): Number of channels in the bottleneck layer. Default: 64.
        num_block (list): Number of dehaze blocks. Default: [2, 2, 2, 2, 2].
    """
    def __init__(self,
                 num_feat: int = 64,
                 max_depth: int = 2,
                 num_block: list = [2, 2, 2, 2, 2]):
        super(IADNet, self).__init__()
        assert len(num_block) == 2 * max_depth + 1, 'num_block should have 2*max_depth+1 elements'
        self.max_depth = max_depth
        self.init_conv = nn.Conv2d(3, num_feat, 3, 1, 1, bias=False)

        def make_layer(in_channels, num_blocks, depth):
            return nn.Sequential(*[IDU(in_channels, depth=depth) for _ in range(num_blocks)])

        # encoder
        feat_temp = num_feat
        self.encoder = nn.ModuleList()
        for i in range(max_depth):
            self.encoder.append(make_layer(feat_temp, num_block[i], i))
            self.encoder.append(Downsample(feat_temp))
            feat_temp *= 2
        
        # bottleneck
        self.bottleneck = make_layer(feat_temp, num_block[max_depth], max_depth)

        # decoder
        self.decoder = nn.ModuleList()
        for i in range(max_depth):
            self.decoder.append(Upsample(feat_temp))
            self.decoder.append(nn.Conv2d(int(feat_temp), int(feat_temp // 2), kernel_size=1, bias=False))
            self.decoder.append(make_layer(feat_temp // 2 , num_block[max_depth+i+1], max_depth-i-1))
            feat_temp //= 2

        # output
        self.out_conv = nn.Conv2d(feat_temp, 3, 3, 1, 1, bias=False)

    def forward(self, x):
        _, _, H, W = x.size()
        pad_h = (2**self.max_depth - H % 2**self.max_depth) % 2**self.max_depth
        pad_w = (2**self.max_depth - W % 2**self.max_depth) % 2**self.max_depth
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # init_conv
        feat = self.init_conv(x)

        # encoder
        down_feats = []
        for i in range(self.max_depth):
            feat = self.encoder[2*i](feat)
            down_feats.append(feat)
            feat = self.encoder[2*i+1](feat)

        # bottleneck
        feat = self.bottleneck(feat)

        # decoder
        for i in range(self.max_depth):
            feat = self.decoder[3*i](feat)
            feat = torch.cat([feat, down_feats[self.max_depth-i-1]], dim=1)
            feat = self.decoder[3*i+1](feat)
            feat = self.decoder[3*i+2](feat)

        out = self.out_conv(feat) + x
        
        return out[:, :, :H, :W]