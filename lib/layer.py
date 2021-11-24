import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2 import *


class PartialConv2d(nn.Conv2d):
    """ Partial convolutional layer (Liu et al., ECCV 18) """

    def __init__(self, *args, **kwargs):
        super(PartialConv2d, self).__init__(*args, **kwargs)

        self.register_buffer(
            'sum_kernel', 
            torch.ones(
                self.out_channels, self.in_channels, 
                self.kernel_size[0], self.kernel_size[1]
            )
        )
        self.is_mean_filter = False

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones_like(x)
        mask = mask.float()
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.size(1) == 1:
            mask = mask.repeat(1, x.size(1), 1, 1)

        in_channels = x.size(1)
        weight = self.weight
        sum_kernel = self.sum_kernel
        groups = self.groups
        if self.is_mean_filter:
            weight = weight.expand(in_channels, 1, -1, -1)
            sum_kernel = sum_kernel.expand(in_channels, in_channels, -1, -1)
            groups = in_channels

        x = F.conv2d(
            x * mask, weight, None, 
            stride=self.stride, padding=self.padding, 
            dilation=self.dilation, groups=groups
        )

        with torch.no_grad():
            # update mask
            mask = F.conv2d(
                mask, sum_kernel, bias=None,
                stride=self.stride, padding=self.padding, 
                dilation=self.dilation
            )
            # compute reweighting factor
            kernel_sum = in_channels * self.kernel_size[0] * self.kernel_size[1]
            gain = kernel_sum / (mask + 1e-8)
            mask = torch.clamp(mask, 0, 1)
            gain = gain * mask

        x = gain * x
        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1) * mask
            x = x + bias
        return x, mask


class PartialAvgPool2d(PartialConv2d):
    """ Partial avearge pooling for anti-aliasing. """

    def __init__(self, kernel_size=2):
        super(PartialAvgPool2d, self).__init__(
            1, 1, kernel_size, kernel_size, bias=False,
        )

        self.weight = nn.Parameter(
            torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2, 
            requires_grad=False
        )
        self.is_mean_filter = True

    def forward(self, x, mask=None):
        x, mask = super(PartialAvgPool2d, self).forward(x, mask)
        return x, mask


def get_conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0,
               bias=True, conv='conv', spectral_norm=False):
    if conv == 'conv':
        Conv2d = nn.Conv2d
    elif conv == 'pconv':
        Conv2d = PartialConv2d
    else:
        raise NotImplementedError(
            '[ERROR] invalid 2D convolution: {:s}'.format(conv)
        )
    layer = Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias)
    if spectral_norm:
        return nn.utils.spectral_norm(layer)
    return layer


def get_actv(actv='relu'):
    if actv == 'relu':
        layer = nn.ReLU(inplace=True)
    elif actv == 'leaky_relu':
        layer = nn.LeakyReLU(0.2, inplace=True)
    elif actv == 'prelu':
        layer = nn.PReLU(init=0.2)
    elif actv == 'elu':
        layer = nn.ELU(inplace=True)
    elif actv == 'none' or actv is None:
        layer = nn.Identity()
    else:
        raise NotImplementedError(
            '[ERROR] invalid activation: {:s}'.format(actv)
        )
    return layer


def get_norm(dim, norm='batch'):
    if norm == 'batch':
        layer = nn.BatchNorm2d(dim)
    elif norm == 'instance':
        layer = nn.InstanceNorm2d(dim, affine=False)
    elif norm == 'none' or norm is None:
        layer = nn.Identity()
    else:
        raise NotImplementedError(
            '[ERROR] invalid normalization: {:s}'.format(norm)
        )
    return layer

###############################################################################
""" GCN utilities """

class RadiusGraph(nn.Module):
    """ Sample a subset of points and gather their neighbors within a ball. """

    def __init__(self, k, radius, scale_factor=1):
        """
        Args:
            k (int): number of points to draw from a ball.
            radius (float): ball radius.
            scale_factor (int): down-sampling factor.
        """
        super(RadiusGraph, self).__init__()

        self.k = k
        self.radius = radius
        self.scale_factor = scale_factor
    
    def forward(self, xyz):
        """
        Args:
            xyz (float tensor, (bs, p0, 3)): point coordinates.

        Returns:
            edges (int tensor, (bs, p1, k, 2)): edges to neighbors in the ball.
            is_filled (bool tensor, (bs, p1, k)): a mask for valid indices.
            child_xyz (float tensor, (bs, p1, 3)): (down-sampled) point coordinates.
        """
        with torch.no_grad():
            bs, p = xyz.shape[:2]
            
            if self.scale_factor > 1:
                # down-sample the point cloud
                ctr_idx = furthest_point_sample(
                    xyz, xyz.size(1) // self.scale_factor
                )                                                           # (bs, p1) 
                child_xyz = gather_operation(
                    xyz.transpose(2, 1), ctr_idx
                ).transpose(2, 1)                                           # (bs, p1, 3)
            else:
                ctr_idx = torch.arange(p, device=xyz.device).repeat(bs, 1)  # (bs, p0)
                child_xyz = xyz
            
            # gather neighboring points from within a ball
            nbr_idx, is_filled = ball_query(
                xyz, child_xyz, self.radius, self.k
            )                                                               # (bs, p1, k)
            ctr_idx = ctr_idx.unsqueeze(-1).expand_as(nbr_idx)              # (bs, p1, k)
            edges = torch.stack([nbr_idx, ctr_idx], -1)                     # (bs, p1, k, 2)
        return edges, is_filled, child_xyz


class GraphConv(nn.Module):
    """ Graph convolutional layer (Li et al., ICCV 19) """

    def __init__(self, in_dim, out_dim, scale_factor, radius, k=16,
                 bias=True, aggregate='max', norm=None, actv='relu', res=False,
                 *args, **kwargs):
        super(GraphConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scale_factor = scale_factor

        self.grouper = RadiusGraph(k, radius, scale_factor)

        self.conv = nn.Sequential(
            get_conv2d(in_dim * 2, out_dim, bias=bias),
            get_norm(out_dim, norm),
        )
        self.actv = get_actv(actv)

        assert aggregate in ('max', 'mean'), \
            '[ERROR] invalid aggregation operator: {:s}'.format(aggregate)
        self.aggregate = aggregate

        self.res_conv = None
        if res:
            if scale_factor > 1 or out_dim != in_dim:
                self.res_conv = nn.Conv1d(in_dim, out_dim, 1, bias=False)
            else:
                self.res_conv = nn.Identity()

    def forward(self, xyz, feats):
        raise NotImplementedError('[ERROR] forward pass not implemented')


class GraphConvBlock(nn.Module):
    """ Graph convolutional block """

    def __init__(self, in_dim, out_dim, scale_factor, radius, k=16,
                 bias=True, aggregate='max', norm=None, actv='relu', res=True, 
                 *args, **kwargs):
        super(GraphConvBlock, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scale_factor = scale_factor

        if isinstance(k, int):
            k = [k] * 2
        if isinstance(radius, float):
            radius = [radius] * 2

        self.grouper1 = RadiusGraph(k[0], radius[0], scale_factor)
        if scale_factor > 1 or k[1] != k[0] or radius[1] != radius[0]:
            self.grouper2 = RadiusGraph(k[1], radius[1], 1)
        else:
            self.grouper2 = None

        self.conv1 = nn.Sequential(
            get_conv2d(in_dim * 2, out_dim, bias=bias),
            get_norm(out_dim, norm),
        )
        self.conv2 = nn.Sequential(
            get_conv2d(out_dim * 2, out_dim, bias=bias),
            get_norm(out_dim, norm),
        )
        self.actv = get_actv(actv)

        assert aggregate in ('max', 'mean'), \
            '[ERROR] invalid aggregation operator: {:s}'.format(aggregate)
        self.aggregate = aggregate

        self.res_conv = None
        if res:
            if scale_factor > 1 or out_dim != in_dim:
                self.res_conv = nn.Conv1d(in_dim, out_dim, 1, bias=False)
            else:
                self.res_conv = nn.Identity()

    def forward(self, xyz, feats):
        raise NotImplementedError('[ERROR] forward pass not implemented')


class GraphConvUpBlock(nn.Module):
    """ Graph convolutional Upsampling block """

    def __init__(self, in_dim, out_dim, radius, k=16, up='linear',
                 bias=True, aggregate='max', norm=None, actv='relu', res=True, 
                 *args, **kwargs):
        super(GraphConvUpBlock, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        if isinstance(k, int):
            k = [k] * 2
        if isinstance(radius, float):
            radius = [radius] * 2

        self.grouper1 = RadiusGraph(k[0], radius[0], 1)
        self.grouper2 = RadiusGraph(k[1], radius[1], 1)
        
        self.conv1 = nn.Sequential(
            get_conv2d(in_dim * 2, out_dim, bias=bias),
            get_norm(out_dim, norm),
        )
        self.up = PointUpsample(up)
        self.conv2 = nn.Sequential(
            get_conv2d(out_dim * 2, out_dim, bias=bias),
            get_norm(out_dim, norm),
        )
        self.actv = get_actv(actv)

        assert aggregate in ('max', 'mean'), \
            '[ERROR] invalid aggregation operator: {:s}'.format(aggregate)
        self.aggregate = aggregate

        self.res_conv = None
        if res:
            if out_dim != in_dim:
                self.res_conv = nn.Conv1d(in_dim, out_dim, 1, bias=False)
            else:
                self.res_conv = nn.Identity()

    def forward(self, xyz, parent_xyz, feats):
        raise NotImplementedError('[ERROR] forward pass not implemented')


class EdgeConv(GraphConv):
    """ Edge convolution (Wang et al., TOG 19) """
    
    def __init__(self, *args, **kwargs):
        super(EdgeConv, self).__init__(*args, **kwargs)

    def forward(self, xyz, feats):
        """
        Args:
            xyz (float tensor, (bs, p0, 3)): input point coordinates.
            feats (float tensor, (bs, c0, p0)): input point features.

        Returns:
            xyz (float tensor, (bs, c1, p1)): output point coordinates.
            feats (float tensor, (bs, c1, p1)): output point features.
        """
        edges, is_filled, xyz = self.grouper(xyz)           # (bs, p1, k, 2)
        is_filled = is_filled.unsqueeze(1)                  # (bs, 1, p, k)

        fi = grouping_operation(feats, edges[..., 1])       # (bs, c0, p1, k)
        fj = grouping_operation(feats, edges[..., 0])       # (bs, c0, p1, k)
        f = self.conv(torch.cat([fi, fj - fi], 1))          # (bs, c1, p1, k)

        if self.aggregate == 'max':
            feats = torch.amax(f, -1)                       # (bs, c1, p1)
        else:
            n_filled = is_filled.sum(-1).clamp(min=1)       # (bs, 1, p1)
            feats = (f * is_filled).sum(-1) / n_filled      # (bs, c1, p1)

        if self.res_conv is not None:
            feats = feats + self.res_conv(fi[..., 0])
        feats = self.actv(feats)
        return xyz, feats


class EdgeConvBlock(GraphConvBlock):
    """ Edge convolutional block with residual connection """

    def __init__(self, *args, **kwargs):
        super(EdgeConvBlock, self).__init__(*args, **kwargs)

    def forward(self, xyz, feats):
        # first conv
        edges, is_filled, xyz = self.grouper1(xyz)          # (bs, p1, k, 2)
        is_filled = is_filled.unsqueeze(1)                  # (bs, 1, p, k)

        fi = grouping_operation(feats, edges[..., 1])       # (bs, c0, p1, k)
        fj = grouping_operation(feats, edges[..., 0])       # (bs, c0, p1, k)
        f = self.conv1(torch.cat([fi, fj - fi], 1))         # (bs, c1, p1, k)

        if self.aggregate == 'max':
            feats = torch.amax(f, -1)                       # (bs, c1, p1)
        else:
            n_filled = is_filled.sum(-1).clamp(min=1)       # (bs, 1, p1)
            feats = (f * is_filled).sum(-1) / n_filled      # (bs, c1, p1)
        feats = self.actv(feats)

        # residual connection
        res = None
        if self.res_conv is not None:
            res = self.res_conv(fi[..., 0])

        # second conv
        if self.grouper2 is not None:
            edges, is_filled, xyz = self.grouper2(xyz)      # (bs, p1, k, 2)
            is_filled = is_filled.unsqueeze(1)              # (bs, 1, p, k)

        fi = grouping_operation(feats, edges[..., 1])       # (bs, c0, p1, k)
        fj = grouping_operation(feats, edges[..., 0])       # (bs, c0, p1, k)
        f = self.conv2(torch.cat([fi, fj - fi], 1))         # (bs, c1, p1, k)

        if self.aggregate == 'max':
            feats = torch.amax(f, -1)                       # (bs, c1, p1)
        else:
            n_filled = is_filled.sum(-1).clamp(min=1)       # (bs, 1, p1)
            feats = (f * is_filled).sum(-1) / n_filled      # (bs, c1, p1)
        
        if res is not None:
            feats = feats + res
        feats = self.actv(feats)
        return xyz, feats


class EdgeConvUpBlock(GraphConvUpBlock):

    def __init__(self, *args, **kwargs):
        super(EdgeConvUpBlock, self).__init__(*args, **kwargs)

    def forward(self, xyz, parent_xyz, feats):
        # first conv
        edges, is_filled, xyz = self.grouper1(xyz)          # (bs, p1, k, 2)
        is_filled = is_filled.unsqueeze(1)                  # (bs, 1, p, k)

        fi = grouping_operation(feats, edges[..., 1])       # (bs, c0, p1, k)
        fj = grouping_operation(feats, edges[..., 0])       # (bs, c0, p1, k)
        f = self.conv1(torch.cat([fi, fj - fi], 1))         # (bs, c1, p1, k)

        if self.aggregate == 'max':
            feats = torch.amax(f, -1)                       # (bs, c1, p1)
        else:
            n_filled = is_filled.sum(-1).clamp(min=1)       # (bs, 1, p1)
            feats = (f * is_filled).sum(-1) / n_filled      # (bs, c1, p1)
        feats = self.actv(feats)

        # residual connection
        res = None
        if self.res_conv is not None:
            res = self.res_conv(fi[..., 0])

        # upsampling
        data = torch.cat([feats, res], 1)
        data = self.up(xyz, parent_xyz, data)
        feats, res = data.split((feats.size(1), res.size(1)), 1)
        xyz = parent_xyz

        # second conv
        edges, is_filled, xyz = self.grouper2(xyz)          # (bs, p1, k, 2)
        is_filled = is_filled.unsqueeze(1)                  # (bs, 1, p, k)

        fi = grouping_operation(feats, edges[..., 1])       # (bs, c0, p1, k)
        fj = grouping_operation(feats, edges[..., 0])       # (bs, c0, p1, k)
        f = self.conv2(torch.cat([fi, fj - fi], 1))         # (bs, c1, p1, k)

        if self.aggregate == 'max':
            feats = torch.amax(f, -1)                       # (bs, c1, p1)
        else:
            n_filled = is_filled.sum(-1).clamp(min=1)       # (bs, 1, p1)
            feats = (f * is_filled).sum(-1) / n_filled      # (bs, c1, p1)
        
        if res is not None:
            feats = feats + res
        feats = self.actv(feats)
        return xyz, feats


class MRConv(GraphConv):
    """ Max-relative convlution (Li et al., TPAMI 21) """

    def __init__(self, *args, **kwargs):
        super(MRConv, self).__init__(*args, **kwargs)

    def forward(self, xyz, feats):
        """
        Args:
            xyz (float tensor, (bs, p0, 3)): input point coordinates.
            feats (float tensor, (bs, c0, p0)): input point features.

        Returns:
            xyz (float tensor, (bs, c1, p1)): output point coordinates.
            feats (float tensor, (bs, c1, p1)): output point features.
        """
        edges, is_filled, xyz = self.grouper(xyz)           # (bs, p1, k, 2)
        is_filled = is_filled.unsqueeze(1)                  # (bs, 1, p1, k)

        fi = grouping_operation(feats, edges[..., 1])       # (bs, c0, p1, k)
        fj = grouping_operation(feats, edges[..., 0])       # (bs, c0, p1, k)
        f = fj - fi                                         # (bs, c0, p1, k)

        if self.aggregate == 'max':
            f = torch.amax(f, -1)                           # (bs, c0, p1)
        else:
            n_filled = is_filled.sum(-1).clamp(min=1)       # (bs, 1, p1)
            f = (f * is_filled).sum(-1) / n_filled          # (bs, c0, p1)

        f = torch.cat([fi[..., 0], f], 1).unsqueeze(-1)     # (bs, c0 * 2, p1, 1)
        feats = self.conv(f).squeeze(-1)                    # (bs, c1, p1)

        if self.res_conv is not None:
            feats = feats + self.res_conv(fi[..., 0])
        feats = self.actv(feats)
        return xyz, feats


class MRConvBlock(GraphConvBlock):
    """ Max-relative convlutional block with residual connection """

    def __init__(self, *args, **kwargs):
        super(MRConvBlock, self).__init__(*args, **kwargs)

    def forward(self, xyz, feats):
        """
        Args:
            xyz (float tensor, (bs, p0, 3)): input point coordinates.
            feats (float tensor, (bs, c0, p0)): input point features.

        Returns:
            xyz (float tensor, (bs, c1, p1)): output point coordinates.
            feats (float tensor, (bs, c1, p1)): output point features.
        """
        # first conv
        edges, is_filled, xyz = self.grouper1(xyz)          # (bs, p1, k, 2)
        is_filled = is_filled.unsqueeze(1)                  # (bs, 1, p1, k)

        fi = grouping_operation(feats, edges[..., 1])       # (bs, c0, p1, k)
        fj = grouping_operation(feats, edges[..., 0])       # (bs, c0, p1, k)
        f = fj - fi                                         # (bs, c0, p1, k)

        if self.aggregate == 'max':
            f = torch.amax(f, -1)                           # (bs, c0, p1)
        else:
            n_filled = is_filled.sum(-1).clamp(min=1)       # (bs, 1, p1)
            f = (f * is_filled).sum(-1) / n_filled          # (bs, c0, p1)

        f = torch.cat([fi[..., 0], f], 1).unsqueeze(-1)     # (bs, c0 * 2, p1, 1)
        feats = self.conv1(f).squeeze(-1)                   # (bs, c1, p1)
        feats = self.actv(feats)
        
        # residual connection
        res = None
        if self.res_conv is not None:
            res = self.res_conv(fi[..., 0])

        # second conv
        if self.grouper2 is not None:
            edges, is_filled, xyz = self.grouper2(xyz)      # (bs, p1, k, 2)
            is_filled = is_filled.unsqueeze(1)              # (bs, 1, p1, k)

        fi = grouping_operation(feats, edges[..., 1])       # (bs, c0, p1, k)
        fj = grouping_operation(feats, edges[..., 0])       # (bs, c0, p1, k)
        f = fj - fi                                         # (bs, c0, p1, k)

        if self.aggregate == 'max':
            f = torch.amax(f, -1)                           # (bs, c0, p1)
        else:
            n_filled = is_filled.sum(-1).clamp(min=1)       # (bs, 1, p1)
            f = (f * is_filled).sum(-1) / n_filled          # (bs, c0, p1)

        f = torch.cat([fi[..., 0], f], 1).unsqueeze(-1)     # (bs, c0 * 2, p1, 1)
        feats = self.conv2(f).squeeze(-1)                   # (bs, c1, p1)
        
        if res is not None:
            feats = feats + res
        feats = self.actv(feats)
        return xyz, feats


class MRConvUpBlock(GraphConvUpBlock):

    def __init__(self, *args, **kwargs):
        super(MRConvUpBlock, self).__init__(*args, **kwargs)

    def forward(self, xyz, parent_xyz, feats):
        # first conv
        edges, is_filled, xyz = self.grouper1(xyz)          # (bs, p1, k, 2)
        is_filled = is_filled.unsqueeze(1)                  # (bs, 1, p1, k)

        fi = grouping_operation(feats, edges[..., 1])       # (bs, c0, p1, k)
        fj = grouping_operation(feats, edges[..., 0])       # (bs, c0, p1, k)
        f = fj - fi                                         # (bs, c0, p1, k)

        if self.aggregate == 'max':
            f = torch.amax(f, -1)                           # (bs, c0, p1)
        else:
            n_filled = is_filled.sum(-1).clamp(min=1)       # (bs, 1, p1)
            f = (f * is_filled).sum(-1) / n_filled          # (bs, c0, p1)

        f = torch.cat([fi[..., 0], f], 1).unsqueeze(-1)     # (bs, c0 * 2, p1, 1)
        feats = self.conv1(f).squeeze(-1)                   # (bs, c1, p1)
        feats = self.actv(feats)
        
        # residual connection
        res = None
        if self.res_conv is not None:
            res = self.res_conv(fi[..., 0])

        # upsampling
        data = torch.cat([feats, res], 1)
        data = self.up(xyz, parent_xyz, data)
        feats, res = data.split((feats.size(1), res.size(1)), 1)
        xyz = parent_xyz

        edges, is_filled, xyz = self.grouper2(xyz)          # (bs, p1, k, 2)
        is_filled = is_filled.unsqueeze(1)                  # (bs, 1, p1, k)

        fi = grouping_operation(feats, edges[..., 1])       # (bs, c0, p1, k)
        fj = grouping_operation(feats, edges[..., 0])       # (bs, c0, p1, k)
        f = fj - fi                                         # (bs, c0, p1, k)

        if self.aggregate == 'max':
            f = torch.amax(f, -1)                           # (bs, c0, p1)
        else:
            n_filled = is_filled.sum(-1).clamp(min=1)       # (bs, 1, p1)
            f = (f * is_filled).sum(-1) / n_filled          # (bs, c0, p1)

        f = torch.cat([fi[..., 0], f], 1).unsqueeze(-1)     # (bs, c0 * 2, p1, 1)
        feats = self.conv2(f).squeeze(-1)                   # (bs, c1, p1)
        
        if res is not None:
            feats = feats + res
        feats = self.actv(feats)
        return xyz, feats


class GCNLayer(nn.Module):

    def __init__(self, conv, *args, **kwargs):
        super(GCNLayer, self).__init__()

        if conv == 'edge':
            self.conv = EdgeConv(*args, **kwargs)
        elif conv == 'mr':
            self.conv = MRConv(*args, **kwargs)
        else:
            raise NotImplementedError(
                '[ERROR] invalid GCN layer: {:s}'.format(conv)
            )

    def forward(self, xyz, feats):
        xyz, feats = self.conv(xyz, feats)
        return xyz, feats


class GCNBlock(nn.Module):

    def __init__(self, block, *args, **kwargs):
        super(GCNBlock, self).__init__()

        if block == 'edge':
            self.block = EdgeConvBlock(*args, **kwargs)
        elif block == 'mr':
            self.block = MRConvBlock(*args, **kwargs)
        else:
            raise NotImplementedError(
                '[ERROR] invalid GCN block: {:s}'.format(block)
            )

    def forward(self, xyz, feats, parent_xyz=None):
        xyz, feats = self.block(xyz, feats)
        return xyz, feats


class GCNUpBlock(nn.Module):

    def __init__(self, block, *args, **kwargs):
        super(GCNUpBlock, self).__init__()

        if block == 'edge':
            self.block = EdgeConvUpBlock(*args, **kwargs)
        elif block == 'mr':
            self.block = MRConvUpBlock(*args, **kwargs)
        else:
            raise NotImplementedError(
                '[ERROR] invalid GCN Up block: {:s}'.format(block)
            )

    def forward(self, xyz, feats, parent_xyz):
        xyz, feats = self.block(xyz, parent_xyz, feats)
        return xyz, feats