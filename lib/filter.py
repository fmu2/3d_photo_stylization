import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import PartialConv2d


class PartialMeanFilter(PartialConv2d):
    """ A partial mean filter for filling in missing pixels. """

    def __init__(self, kernel_size=3):
        super(PartialMeanFilter, self).__init__(
            1, 1, kernel_size, 1, kernel_size // 2, bias=False,
        )

        self.weight = nn.Parameter(
            torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2, 
            requires_grad=False
        )
        self.is_mean_filter = True

    def forward(self, x, mask=None):
        tmp, new_mask = super(PartialMeanFilter, self).forward(x, mask)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            x = x * mask + tmp * ~mask
        else:
            x = tmp
        return x, new_mask


class MedianFilter(nn.Module):
    """ A median filter for filling in missing pixels. """
    
    def __init__(self, kernel_size=3):
        super(MedianFilter, self).__init__()

        assert kernel_size % 2 == 1, 'kernel size must be odd'

        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        tmp = F.pad(x, [self.pad] * 4, mode='replicate')
        tmp = F.unfold(tmp, self.kernel_size)
        tmp = tmp.reshape(b, c, -1, h * w)
        tmp = tmp.median(2)[0].reshape(b, c, h, w)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1) 
            x = x * mask + tmp * ~mask
        else:
            x = tmp
        return x