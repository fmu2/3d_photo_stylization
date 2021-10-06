import torch
import torch.nn as nn

from .utils import *


class PointNetSAModule(nn.Module):
    """ PointNet set abstraction layer """

    def __init__(self, in_dim, out_dim, k=16, radius=None, 
                 scale_factor=1, bias=True, 
                 aggregate='max', norm=None, actv='relu', res=False):
        super(PointNetSAModule, self).__init__()

        if isinstance(k, (list, tuple)):
            assert isinstance(radius, (list, tuple)) and len(radius) == len(k), \
                ('[ERROR] k {:d} and radius {:d} must have the same length'
                 ''.format(len(k), len(radius))
                )
        else:
            k, radius = [k], [radius]
        n_scales = len(k)

        assert aggregate in ('max', 'mean'), \
            '[ERROR] invalid aggregation operator: {:s}'.format(aggregate)
        self.aggregate = aggregate

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for i in range(n_scales):
            self.groupers.append(QueryAndGroup(radius[i], k[i]))
            mlps = nn.Sequential(
                nn.Conv1d(in_dim, out_dim, 1, bias=bias),
                get_norm(out_dim, norm), get_actv(actv),
                nn.Conv1d(out_dim, out_dim, 1, bias=bias),
                get_norm(out_dim, norm), get_actv(actv),
            )
            self.mlps.append(mlps)

        if n_scales > 1:
            self.zip = nn.Conv1d(out_dim * n_scales, out_dim, 1, bias=False)
        else:
            self.zip = nn.Identity()

        self.res_conv = None
        if res:
            if out_dim != in_dim:
                self.res_conv = nn.Conv1d(in_dim, out_dim, 1, bias=False)
            else:
                self.res_conv = nn.Identity()

    def forward(self, xyz, feats):
        """
        Args:
            xyz (float tensor, (bs, n, 3)): point coordinates.
            feats (float tensor, (bs, c0, n)): point features.

        Returns:
            child_xyz (float tensor, (bs, m, 3)): child point coordinates.
            child_feats (float tensor, (bs, c1, m)): child point features.
        """
        child_xyz = xyz
        if self.scale_factor > 1:
            n_points = xyz.size(1) // self.scale_factor
            child_idx = furthest_point_sample(xyz, n_points)                # (bs, m)
            child_xyz = gather_operation(
                    xyz.transpose(2, 1), child_idx
                ).transpose(2, 1)                                           # (bs, m, 3)

        child_feats_list = []
        for i in range(len(self.groupers)):
            child_feats = self.mlps[i](feats)                               # (bs, c1, n)
            child_feats, is_filled = self.groupers[i](
                xyz, child_xyz, child_feats
            )                                                               # (bs, c1, m, s)
            if self.aggregate == 'max':
                child_feats = torch.max(child_feats, -1)                    # (bs, c1, m)
            else:
                is_filled = is_filled.unsqueeze(1)                          # (bs, 1, m, s)
                n_filled = is_filled.sum(-1).clamp(min=1)                   # (bs, 1, m)
                child_feats = (child_feats * is_filled).sum(-1) / n_filled  # (bs, c1, m)
            child_feats_list.append(child_feats)
        child_feats = torch.cat(child_feats_list, 1)
        
        child_feats = self.zip(child_feats)
        if self.res_conv is not None:
            feats = gather_operation(feats, child_idx)
            child_feats = child_feats + self.res_conv(feats)
        return child_xyz, child_feats


class PointNetFPModule(nn.Module):
    """ PointNet feature propagation layer """

    def __init__(self, in_dim, skip_dim, out_dim, 
                 bias=True, norm=None, actv='relu', res=False):
        super(PointNetFPModule, self).__init__()

        self.mlps = nn.Sequential(
            nn.Conv1d(in_dim + skip_dim, out_dim, 1, bias=bias),
            get_norm(out_dim, norm), get_actv(actv),
            nn.Conv1d(out_dim, out_dim, 1, bias=bias),
            get_norm(out_dim, norm), get_actv(actv),
        )

        self.res_conv = None
        if res:
            if out_dim != in_dim:
                self.res_conv = nn.Conv1d(in_dim, out_dim, 1, bias=False)
            else:
                self.res_conv = nn.Identity()

    def forward(self, xyz, parent_xyz, feats, skip_feats):
        """
        Args:
            xyz (float tensor, (bs, m, 3)): point coordinates.
            parent_xyz (float tensor, (bs, n, 3)): parent point coordinates.
            feats (float tensor, (bs, c0, m)): point features.
            skip_feats (float tensor, (bs, c1, n)): parent point features 
                passed along skip connection.
        
        Returns:
            parent_feats (float tensor, (bs, c2, n)): parent point features.
        """
        dist, idx = three_nn(parent_xyz, xyz)                   # (bs, n, 3)
        inv_dist = 1. / (dist + 1e-8)                           # (bs, n, 3)
        norm = torch.sum(inv_dist, 2, keepdim=True)             # (bs, n, 1)
        weight = inv_dist / norm                                # (bs, n, 3)
        feats = three_interpolate(feats, idx, weight)           # (bs, c0, n)
        
        parent_feats = torch.cat([feats, skip_feats], 1)
        parent_feats = self.mlps(parent_feats)

        if self.res_conv is not None:
            parent_feats = parent_feats + self.res_conv(feats)
        return parent_feats


class PointDownsample(nn.Module):
    """ Point downsampling layer """
    
    def __init__(self, k=None, radius=None, scale_factor=2, mode='max'):
        """
        Args:
            k (int): number of points to draw from a ball.
            radius (float): ball radius.
            scale_factor (int): down-sampling factor.
            mode (str): pooling method ('max', 'mean', 'linear').
        """
        super(PointDownsample, self).__init__()

        assert mode in ('max', 'mean', 'linear'), \
            '[ERROR] invalid down-sampling operator: {:s}'.format(mode)
        self.mode = mode
        
        if mode == 'linear':
            self.grouper = None
        else:
            self.grouper = QueryAndGroup(radius, k)

        self.scale_factor = scale_factor

    def forward(self, xyz, feats, child_xyz=None):
        """
        Args:
            xyz (float tensor, (bs, n, 3)): current point coordinates.
            feats (float tensor, (bs, c0, n)): current point features.
            child_xyz ((optional) float tensor, (bs, m, 3)): child point coordinates.

        Returns:
            child_xyz (float tensor, (bs, m, 3)): child point coordinates.
            child_feats (float tensor, (bs, c1, m)): child point features.
        """
        assert xyz.size(1) == feats.size(2), \
            ('[ERROR] number of points ({:d}) and features ({:d}) must match '
             ''.format(xyz.size(1), feats.size(2))
            )

        if child_xyz is None:
            child_xyz = xyz
            if self.scale_factor > 1:
                n_points = xyz.size(1) // self.scale_factor
                child_idx = furthest_point_sample(xyz, n_points)            # (bs, m)
                child_xyz = gather_operation(
                    xyz.transpose(2, 1), child_idx
                ).transpose(2, 1)                                           # (bs, m, 3)

        if self.mode == 'linear':
            dist, idx = three_nn(child_xyz, xyz)
            inv_dist = 1 / (dist + 1e-8)
            norm = torch.sum(inv_dist, 2, keepdim=True)
            weight = inv_dist / norm
            child_feats = three_interpolate(feats, idx, weight)
        else:
            child_feats, is_filled = self.grouper(xyz, child_xyz, feats)    # (bs, c, m, s)
            if self.mode == 'max':
                child_feats = torch.max(child_feats, -1)                    # (bs, c, m)
            elif self.mode == 'mean':
                is_filled = is_filled.unsqueeze(1)                          # (bs, 1, m, s)
                n_filled = is_filled.sum(-1).clamp(min=1)                   # (bs, 1, m)
                child_feats = (child_feats * is_filled).sum(-1) / n_filled  # (bs, c, m)
        return child_xyz, child_feats


class PointUpsample(nn.Module):
    """ Point upsampling layer """

    def __init__(self, mode='linear'):
        super(PointUpsample, self).__init__()

        assert mode in ('nearest', 'linear'), \
            '[ERROR] invalid up-sampling operator: {:s}'.format(mode)
        self.mode = mode

    def forward(self, xyz, parent_xyz, feats):
        """
        Args:
            xyz (float tensor, (bs, m, 3)): current point coordinates.
            parent_xyz (float tensor, (bs, n, 3)): parent point coordinates.
            feats (float tensor, (bs, c, m)): current point features.
        
        Returns:
            parent_feats (float tensor, (bs, c, n)): parent point features.
        """
        dist, idx = three_nn(parent_xyz, xyz)
        if self.mode == 'linear':
            inv_dist = 1 / (dist + 1e-8)
            norm = torch.sum(inv_dist, 2, keepdim=True)
            weight = inv_dist / norm
            parent_feats = three_interpolate(feats, idx, weight)
        else:
            bs, c, m = feats.shape
            n = idx.size(1)
            idx = idx[..., 0]
            idx = idx + torch.arange(bs, device=idx.device).view(-1, 1) * m
            idx = idx.flatten()
            feats = feats.transpose(2, 1).flatten(0, 1)
            parent_feats = feats[idx].reshape(bs, n, c).transpose(2, 1)
        return parent_feats