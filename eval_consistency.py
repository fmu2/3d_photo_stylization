import os
import math
import random
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import imageio
from PIL import Image

from lib.module import Unprojector, ViewTransformer, Renderer
from lib.render import view2ndc, is_visible
from lib.metric import RMSE, LPIPS
from lib.util import *


def main(args):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load LDI
    try:
        ldi = sio.loadmat(args.ldi_path)
        fov, h, w = float(ldi['fov']), int(ldi['h']), int(ldi['w'])
        uv = ldi['uv'].astype(np.float32) + 0.5
        z = ldi['z'].astype(np.float32)
        print('LDI loaded')
    except:
        raise IOError(
            '[ERROR] LDI loading failed: {:s}'.format(args.ldi_path)
        )
    uv = torch.from_numpy(uv)                       # (p, 2)
    z = torch.from_numpy(z)                         # (p)

    # camera intrinsics
    fov = math.radians(fov)
    fx = fy = 0.5 * h * math.tan((math.pi - fov) / 2)
    cx, cy = w / 2, h / 2
    K = torch.Tensor([fov, fx, fy, cx, cy])         # (5,)

    uv = uv[None].cuda()                            # (1, p, 2)
    z = z[None].cuda()                              # (1, p, 1)
    K = K[None].cuda()                              # (1, 5)

    # prepare point cloud
    unprojector = Unprojector()
    xyz = unprojector(uv, z, K)                     # (1, p, 3)

    # load video frames and camera parameters
    ims = []
    vid = imageio.get_reader(
        os.path.join(args.vid_path, 'video.mp4'), 'ffmpeg'
    )
    for _, im in enumerate(vid):
        scale = round(h / im.shape[0])
        im = Image.fromarray(im)
        im = im.resize((w // scale, h // scale))
        im = np.array(im)
        ims.append(im)

    ims = np.stack(ims) / 255.                      # (v, h, w, 3)
    ims = torch.from_numpy(ims.astype(np.float32))
    ims = ims.permute(0, 3, 1, 2)                   # (v, 3, h, w)

    views = sio.loadmat(os.path.join(args.vid_path, 'views.mat'))
    Ms = torch.from_numpy(views['Ms'].astype(np.float32))
    fovs = torch.from_numpy(views['fovs'][0].astype(np.float32))
    n_views = len(Ms)

    ims = ims[None].cuda()                          # (1, v, 3, h, w)
    Ms = Ms[None].cuda()                            # (1, v, 3, 4)
    fovs = fovs[None].cuda()                        # (1, v)

    view_transformer = ViewTransformer()

    anti_aliasing = h // ims.size(-2)
    renderer = Renderer(anti_aliasing=anti_aliasing)

    rmse_fn, lpips_fn = RMSE().cuda(), LPIPS().cuda()
    rmse_meter, lpips_meter = AverageMeter(), AverageMeter()

    for i in range(args.n_pairs):
        v1, v2 = random.sample(range(n_views), 2)
        im1, im2 = ims[:, v1], ims[:, v2]
        fov1, fov2 = fovs[:, v1], fovs[:, v2]
        M1, M2 = Ms[:, v1], Ms[:, v2]
        
        ## FIRST VIEW
        new_xyz = view_transformer(xyz, M1)

        # view space -> NDC space
        ## NOTE: assume that valid points have depth < 1e5.
        z = new_xyz[..., 2]
        near = 0.99 * z.amin(1)
        far = (z * (z < 1e5)).quantile(0.95, 1)
        far = torch.maximum(far, near * 2)
        new_xyz_ndc = view2ndc(new_xyz, near, far, fov1, w / h)

        # filter out invisible points
        _, viz = is_visible(new_xyz_ndc, h, w)
        xyz = xyz[:, viz[0]]
        new_xyz_ndc = new_xyz_ndc[:, viz[0]]

        # sample RGB values from first view
        uv = new_xyz_ndc[..., :2]
        rgb1 = F.grid_sample(im1, uv.unsqueeze(-2), align_corners=False)
        rgb1 = rgb1.flatten(2)

        ## SECOND VIEW
        new_xyz = view_transformer(xyz, M2)

        # view space -> NDC space
        ## NOTE: assume that valid points have depth < 1e5.
        z = new_xyz[..., 2]
        near = 0.99 * z.amin(1)
        far = (z * (z < 1e5)).quantile(0.95, 1)
        far = torch.maximum(far, near * 2)
        new_xyz_ndc = view2ndc(new_xyz, near, far, fov2, w / h)

        # filter out invisible points
        _, viz = is_visible(new_xyz_ndc, h, w)
        new_xyz = new_xyz[:, viz[0]]
        rgb1 = rgb1[..., viz[0]]
        new_xyz_ndc = new_xyz_ndc[:, viz[0]]

        # sample RGB values from second view
        uv = new_xyz_ndc[..., :2]
        rgb2 = F.grid_sample(im2, uv.unsqueeze(-2), align_corners=False)
        rgb2 = rgb2.flatten(2)

        # rasterize
        out_dict = renderer(
            new_xyz, rgb2, fov2, h // anti_aliasing, w // anti_aliasing
        )
        pred, mask = out_dict['data'], out_dict['mask']
        tgt = im2 * mask

        # evaluate metrics
        rmse = rmse_fn(rgb1, rgb2)
        lpips = lpips_fn(pred, tgt, mask)
        rmse_meter.update(rmse.item())
        lpips_meter.update(lpips.item())

        # save warp-target pair
        pred = pred[0].permute(1, 2, 0).cpu().numpy()
        tgt = tgt[0].permute(1, 2, 0).cpu().numpy()
        out = np.concatenate((pred, tgt), 1)
        out = Image.fromarray((out * 255).astype(np.uint8))
        out.save(os.path.join(save_path, '{:03d}.png'.format(i + 1)))

    print('mean rmse: {:.3f}'.format(rmse_meter.item()))
    print('mean lpips: {:.3f}'.format(lpips_meter.item()))

###############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=str, default='0', 
                        help='GPU device ID')

    parser.add_argument('-ldi', '--ldi_path', type=str, 
                        help='LDI path')
    parser.add_argument('-vid', '--vid_path', type=str,
                        help='video path')

    parser.add_argument('-p', '--n_pairs', type=int, default=100,
                        help='number of image pairs')
    
    args = parser.parse_args()

    check_file(args.ldi_path)
    check_path(args.vid_path)

    # set up save folder
    save_path = os.path.join(args.vid_path, 'warps')
    ensure_path(save_path)

    set_gpu(args.gpu)

    main(args)