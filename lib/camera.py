import math

import torch


def make_circle(n_frames, x_lim, y_lim, z_lim):
    if isinstance(x_lim, (list, tuple)):
        x_lim = x_lim[-1]
    if isinstance(y_lim, (list, tuple)):
        y_lim = y_lim[-1]
    if not isinstance(z_lim, (list, tuple)):
        z_lim = (-z_lim, z_lim)
    assert len(z_lim) == 2, \
        'z_lim must have two values, got {:d}'.format(len(z_lim))

    R = torch.eye(3)
    tics = torch.linspace(-2, 2, n_frames)
    xs = torch.cos(tics * math.pi) * x_lim
    ys = torch.sin(tics * math.pi) * y_lim
    dz = z_lim[1] - z_lim[0]
    zs = (torch.cos(tics * math.pi / 2) + 1) * dz / 2 + z_lim[0]
    ts = torch.stack([-xs, -ys, -zs], -1).unsqueeze(-1)
    Ms = torch.cat([R.repeat(n_frames, 1, 1), ts], -1)
    return Ms


def make_swing(n_frames, x_lim, z_lim):
    return make_circle(n_frames, x_lim, 0, z_lim)


def make_ken_burns(n_frames, x_lim, y_lim, z_lim):
    if not isinstance(x_lim, (list, tuple)):
        x_lim = (-x_lim, x_lim)
    if not isinstance(y_lim, (list, tuple)):
        y_lim = (-y_lim, y_lim)
    if not isinstance(z_lim, (list, tuple)):
        z_lim = (-z_lim, z_lim)
    assert len(x_lim) == 2, \
        'x_lim must have two values, got {:d}'.format(len(x_lim))
    assert len(y_lim) == 2, \
        'y_lim must have two values, got {:d}'.format(len(y_lim))
    assert len(z_lim) == 2, \
        'z_lim must have two values, got {:d}'.format(len(z_lim))

    R = torch.eye(3)
    xs = torch.linspace(x_lim[0], x_lim[1], n_frames)
    ys = torch.linspace(y_lim[0], y_lim[1], n_frames)
    zs = torch.linspace(z_lim[0], z_lim[1], n_frames)
    ts = torch.stack([-xs, -ys, -zs], -1).unsqueeze(-1)
    Ms = torch.cat([R.repeat(n_frames, 1, 1), ts], -1)
    return Ms


def make_zoom(n_frames, z_lim):
    return make_ken_burns(n_frames, 0, 0, z_lim)


def make_dolly_zoom(n_frames, z_lim, fov, ctr_depth):
    Ms = make_zoom(n_frames, z_lim)
    plane_width = math.tan(fov / 2) * ctr_depth
    ctr_depths = ctr_depth + Ms[:, 2, 3]
    fovs = 2 * torch.atan2(plane_width, ctr_depths)
    return Ms, fovs