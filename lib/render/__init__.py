import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *


class Splatter(nn.Module):

    def __init__(self):
        super(Splatter, self).__init__()

    def forward(self, xyz, data, h, w):
        """
        Args:
            xyz (float tensor, (bs, p, 3)): point coordinates in NDC space.
            data (float tensor, (bs, c, p)): point-associated data.
            h (int): height of output data map (unit: px).
            w (int): width of output data map (unit: px).

        Returns:
            data (float tensor, (bs, c, h, w)): rendered data maps.
            conf (float tensor, (bs, 1, h, w)): per-pixel confidence map.
            viz (bool tensor, (bs, p)): mask for visible points.
        """
        assert xyz.size(1) == data.size(2), \
            ('ERROR: point cloud size ({:d}) and number of features ({:d}) '
             'must match'.format(xyz.size(1), data.size(2))
            )
        
        data, conf, viz = render_point_cloud(xyz, data, h, w)
        return data, conf, viz
