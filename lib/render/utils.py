import torch
import torch.nn as nn
from torch.autograd import Function

try:
    from . import ops
except ImportError:
    raise ImportError(
        "Failed to import lib.render.ops module.\n"
        "Run 'python setup_render.py build_ext --inplace' first."
    )

__all__ = ['view2ndc', 'ndc2view', 'render_point_cloud', 'is_visible']


class RenderPointCloud(Function):
    @staticmethod
    def forward(ctx, xyz, data, h, w):
        """
        Args:
            xyz (float tensor, (bs, p, 3)): 3D point coordinates.
            data (float tensor, (bs, c + 1, p)): point-associated data.
            h (int): height of depth map.
            w (int): width of depth map.

        Returns:
            out (float tensor, (bs, c, h, w)): rendered feature maps.
            is_visible (bool tensor, (bs, p)): mask for visible points.
        """
        assert xyz.is_cuda, 'xyz must be a GPU tensor'
        assert data.is_cuda, 'data must be a GPU tensor'

        xyz = xyz.contiguous()
        data = data.contiguous()

        z_buffer = ops.rasterize(xyz, h, w)
        ops.refine_z_buffer(z_buffer)
        out, is_visible = ops.splat(data, xyz, z_buffer)

        ctx.save_for_backward(xyz, z_buffer)
        ctx.mark_non_differentiable(is_visible)

        return out, is_visible

    @staticmethod
    def backward(ctx, grad_out, grad_is_visible=None):
        assert grad_out.is_cuda, 'grad_out must be a GPU tensor'

        xyz, z_buffer = ctx.saved_tensors

        grad_out = grad_out.contiguous()

        grad_data = ops.splat_grad(grad_out, xyz, z_buffer)

        return None, grad_data, None, None, None


def view2ndc(xyz, near, far, fov, ar=1):
    """
    View space -> NDC space.

    Args:
        xyz (float tensor, (bs, p, 3)): xyz coordinates in view space.
        near (float tensor, (bs,)): near bound.
        far (float tensor, (bs,)): far bound.
        fov (float tensor, (bs,)): vertical angular field of view.
        ar (float): aspect ratio (width / height).

    Returns:
        xyz (float tensor, (bs, p, 3)): xyz coordinates in NDC space.
    """
    bs = xyz.size(0)
    tmp1 = 1 / torch.tan(fov / 2)                           # (bs,)
    tmp2 = (far + near) / (far - near)                      # (bs,)
    tmp3 = 2 * far * near / (far - near)                    # (bs,)

    tmp1 = tmp1.view(bs, 1, 1)
    tmp2 = tmp2.view(bs, 1, 1)
    tmp3 = tmp3.view(bs, 1, 1)

    scale = torch.cat([tmp1 / ar, tmp1, tmp2], -1)          # (bs, 1, 3)
    z = xyz[..., 2:]                                        # (bs, p, 1)
    xyz = scale * xyz                                       # (bs, p, 3)
    xyz[..., 2:] = xyz[..., 2:] - tmp3
    xyz = xyz / z
    return xyz


def ndc2view(xyz, near, far, fov, ar=1):
    """
    NDC space -> view space.

    Args:
        xyz (float tensor, (bs, p, 3)): xyz coordinates in NDC space.
        near (float tensor, (bs,)): near bound.
        far (float tensor, (bs,)): far bound.
        fov (float tensor, (bs,)): vertical angular field of view.
        ar (float): aspect ratio (width / height).

    Returns:
        xyz (float tensor, (bs, p, 3)): xyz coordinates in view space.
    """
    bs = xyz.size(0)
    tmp1 = 2 * far * near / (far - near)                    # (bs,)
    tmp2 = (far + near) / (far - near)                      # (bs,)
    tmp3 = torch.tan(fov / 2)                               # (bs,)

    tmp1 = tmp1.view(bs, 1, 1)
    tmp2 = tmp2.view(bs, 1, 1)
    tmp3 = tmp3.view(bs, 1, 1)

    z = tmp1 / (tmp2 - xyz[..., 2:])
    xyz[..., :1] = xyz[..., :1] * ar
    xyz[..., :2] = z * xyz[..., :2] * tmp3
    xyz[..., 2:] = z
    return xyz


def render_point_cloud(xyz, data, h, w):
    """
    Render a point cloud.
    
    Args:
        xyz (float tensor, (bs, p, 3)): 3D point coordinates.
        data (float tensor, (bs, c, p)): point-associated data.
        h (int): height of rendered output.
        w (int): width of rendered output.
        refine_z_buffer (bool): if True, guard against shine-through effect.

    Returns:
        render (float tensor, (bs, c, h, w)): rendered data maps.
        conf (float tensor, (bs, 1, h, w)): confidence map.
        is_visible (bool tensor, (bs, p)): mask for visible points.
    """
    data = torch.cat([data, data.new_ones(data.size(0), 1, data.size(2))], 1)
    
    out, is_visible = RenderPointCloud.apply(xyz, data, h, w)
    
    render = out[:, :-1] / (out[:, -1:] + 1e-8)             # (bs, c, h, w)
    conf = out[:, -1:]                                      # (bs, 1, h, w)
    return render, conf, is_visible


def is_visible(xyz, h, w):
    """
    Check visibility of input points.

    Args:
        xyz (float tensor, (bs, p, 3)): 3D point coordinates.
        h (int): height of rendered output.
        w (int): width of rendered output.
    
    Returns:
        conf (float tensor, (bs, 1, h, w)): confidence map.
        is_visible (bool tensor, (bs, p)): mask for visible points.
    """
    data = xyz.new_ones(xyz.size(0), 1, xyz.size(1))
    conf, is_visible = RenderPointCloud.apply(xyz, data, h, w)
    return conf, is_visible