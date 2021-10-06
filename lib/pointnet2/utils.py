import torch
import torch.nn as nn
from torch.autograd import Function

try:
    from . import ops
except ImportError:
    raise ImportError(
        "Failed to import lib.pointnet2.ops module.\n"
        "Run 'python setup_pointnet2.py build_ext --inplace' first."
    )


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, n_points):
        """
        Args:
            xyz (float tensor, (bs, n, 3)): parent point coordinates. 
            n_points (int): size of child point set (m).
        
        Returns:
            idx (int tensor, (bs, m)): indices of sampled points.
        """
        assert xyz.is_cuda, 'xyz must be a GPU tensor'
        assert xyz.size(1) >= n_points, \
            'the full set must be no smaller than sample size'

        xyz = xyz.contiguous()
        
        idx = ops.furthest_point_sampling(xyz, n_points)
        
        ctx.mark_non_differentiable(idx)
        
        return idx

    @staticmethod
    def backward(ctx, grad_idx=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        """
        Args:
            features (float tensor, (bs, c, n)): parent point features.
            idx (int tensor, (bs, m)): indices of sampled points.
        
        Returns:
            out (float tensor, (bs, c, m)): sampled point features.
        """
        assert features.is_cuda, 'features must be a GPU tensor'
        assert idx.is_cuda, 'idx must be a GPU tensor'
        
        features = features.contiguous()
        idx = idx.int().contiguous()

        out = ops.gather_points(features, idx)

        ctx.save_for_backward(idx)
        ctx.n = features.size(2)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        assert grad_out.is_cuda, 'grad_out must be a GPU tensor'

        idx, = ctx.saved_tensors
        n = ctx.n

        grad_out = grad_out.contiguous()

        grad_features = ops.gather_points_grad(grad_out, idx, n)
        
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, parent_xyz, child_xyz):
        """
        Args:
            parent_xyz (float tensor, (bs, n, 3)): parent point coordinates.
            child_xyz (float tensor, (bs, m, 3)): child point coordinates.
        
        Returns:
            dist (float tensor, (bs, n, 3)): Eucledian distances from a parent 
                point to its three nearest child points.
            idx (int tensor, (bs, n, 3)): indices of the nearest child points.
        """
        assert parent_xyz.is_cuda, 'parent_xyz must be a GPU tensor'
        assert child_xyz.is_cuda, 'child_xyz must be a GPU tensor'

        parent_xyz = parent_xyz.contiguous()
        child_xyz = child_xyz.contiguous()

        dist, idx = ops.three_nn(parent_xyz, child_xyz)

        ctx.mark_non_differentiable(dist)
        ctx.mark_non_differentiable(idx)
        
        return dist, idx

    @staticmethod
    def backward(ctx, grad_dist=None, grad_idx=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        """
        Args:
            features (float tensor, (bs, c, m)): child point features.
            idx (int tensor, (bs, n, 3)): parent point coordinates.
            weight (float tensor, (bs, m, 3)): interpolation weights.
        
        Returns:
            out (float tensor, (bs, c, n)): interpolated parent point features.
        """
        assert features.is_cuda, 'features must be a GPU tensor'
        assert idx.is_cuda, 'idx must be a GPU tensor'
        assert weight.is_cuda, 'weight must be a GPU tenor'

        features = features.contiguous()
        idx = idx.int().contiguous()
        weight = weight.contiguous()

        out = ops.three_interpolate(features, idx, weight)

        ctx.save_for_backward(idx, weight)
        ctx.m = features.size(2)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        assert grad_out.is_cuda, 'grad_out must be a GPU tensor'

        idx, weight = ctx.saved_tensors
        m = ctx.m

        grad_out = grad_out.contiguous()

        grad_features = ops.three_interpolate_grad(grad_out, idx, weight, m)
        
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        """
        Args:
            features (float tensor, (bs, c, n)): parent point features.
            idx (int tensor, (bs, m, s)): indices of sampled points for each ball.
        
        Returns:
            out (float tensor, (bs, c, m, s)): grouped parent point features.
        """
        assert features.is_cuda, 'features must be a GPU tensor'
        assert idx.is_cuda, 'idx must be a GPU tensor'

        features = features.float().contiguous()
        idx = idx.int().contiguous()

        out = ops.group_points(features, idx)

        ctx.save_for_backward(idx)
        ctx.n = features.size(2)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        assert grad_out.is_cuda, 'grad_out must be a GPU tensor'

        idx, = ctx.saved_tensors
        n = ctx.n

        grad_out = grad_out.contiguous()

        grad_features = ops.group_points_grad(grad_out, idx, n)
        
        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, parent_xyz, child_xyz, radius, k):
        """
        Args:
            parent_xyz (float tensor, (bs, n, 3)): parent point coordinates.
            child_xyz (float tensor, (bs, m, 3)): child point coordinates.
            radius (float): ball radius.
            k (int): maximum number of points to draw from each ball (s).

        Returns:
            idx (int tensor, (bs, m, s)): indices of sampled points for each ball.
            is_filled (bool tensor, (bs, m, s)): a mask for valid indices.
        """
        assert parent_xyz.is_cuda, 'parent_xyz must be a GPU tensor'
        assert child_xyz.is_cuda, 'child_xyz must be a GPU tensor'

        parent_xyz = parent_xyz.contiguous()
        child_xyz = child_xyz.contiguous()

        idx, is_filled = ops.ball_query(parent_xyz, child_xyz, radius, k)

        ctx.mark_non_differentiable(idx)
        ctx.mark_non_differentiable(is_filled)

        return idx, is_filled

    @staticmethod
    def backward(ctx, grad_idx=None, grad_is_filled=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius, k, use_xyz=False):
        """
        Args:
            radius (float): ball radius.
            k (int): maximum number of points to draw from each ball.
            use_xyz (bool): if True, include point coordinates as features.
        """
        super(QueryAndGroup, self).__init__()

        self.radius = radius
        self.k = k
        self.use_xyz = use_xyz

    def forward(self, xyz, child_xyz, feats=None):
        """
        Args:
            xyz (float tensor, (bs, n, 3)): current point coordinates.
            child_xyz (float tensor, (bs, m, 3)): child point coordinates.
            feats (float tensor, (bs, c, n)): parent point features.
        
        Returns:
            grouped_feats (float tensor, (bs, c(+3), m, s)): grouped point features.
            is_filled (bool tensor, (bs, m, s)): a mask for valid indices.
        """
        idx, is_filled = ball_query(xyz, child_xyz, self.radius, self.k)    # (bs, m, s)
        grouped_xyz = grouping_operation(xyz.transpose(1, 2), idx)          # (bs, 3, m, s)
        grouped_xyz -= child_xyz.transpose(1, 2).unsqueeze(-1)              # (bs, 3, m, s)

        if feats is not None:
            grouped_feats = grouping_operation(feats, idx)                  # (bs, c, m, s)
            if self.use_xyz:
                grouped_feats = torch.cat([grouped_xyz, grouped_feats], 1)  # (bs, c+3, m, s)
        else:
            assert self.use_xyz, "coordinates are the only available features"
            grouped_feats = grouped_xyz

        return grouped_feats, is_filled


class GroupAll(nn.Module):
    def __init__(self, use_xyz=True):
        super(GroupAll, self).__init__()
        
        self.use_xyz = use_xyz

    def forward(self, parent_xyz, child_xyz=None, feats=None):
        """
        Args:
            parent_xyz (float tensor, (bs, n, 3)): parent point coordinates.
            child_xyz: ignored.
            feats (float tensor, (bs, c, n)): parent point features.
        
        Returns:
            grouped_feats (float tensor, (bs, c(+3), 1, n)): grouped point features.
            is_filled (bool tensor, (bs, 1, n)): a mask for valid indices.
        """
        grouped_xyz = parent_xyz.transpose(1, 2).unsqueeze(2)               # (bs, 3, 1, n)
        if feats is not None:
            grouped_feats = feats.unsqueeze(2)                              # (bs, c, 1, n)
            if self.use_xyz:
                grouped_feats = torch.cat([grouped_xyz, grouped_feats], 1)
        else:
            assert self.use_xyz, "coordinates are the only available features"
            grouped_feats = grouped_xyz

        is_filled = grouped_feats.new_ones(
            grouped_feats.size(0), 1, grouped_feats.size(-1)
        ).bool()

        return grouped_feats, is_valid


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
        layer = nn.BatchNorm1d(dim)
    elif norm == 'instance':
        layer = nn.InstanceNorm1d(dim, affine=False)
    elif norm == 'none' or norm is None:
        layer = nn.Identity()
    else:
        raise NotImplementedError(
            '[ERROR] invalid normalization: {:s}'.format(norm)
        )
    return layer