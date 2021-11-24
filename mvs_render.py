import os
import math
import time
import random
import argparse

import torch
import numpy as np
import scipy.io as sio
import imageio

from lib.module import Unprojector, ViewTransformer, Renderer
from lib.util import *


def main(args):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load data
    try:
        data = sio.loadmat(args.data_path)
        fov = float(data['fov'])
        rgb = data['rgb'].astype(np.float32) / 255
        z = data['z'].astype(np.float32)
        h, w = z.shape[-2:]
        Ms, views = data['Ms'], data['views']       # input / output views
        ctr_idx = len(Ms) // 2                      # index of center view

        n_in_views, n_out_views = len(Ms), len(views)
        rgb = rgb.reshape(n_in_views, -1, 3)
        z = z.reshape(n_in_views, -1)
        uv = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1)
        uv = uv.astype(np.float32) + 0.5
        uv = uv.reshape(-1, 2)

        print('data loaded')
    except:
        raise IOError(
            '[ERROR] data loading failed: {:s}'.format(args.data_path)
        )

    rgb = torch.from_numpy(rgb)                     # (vi, p, 3)
    uv = torch.from_numpy(uv)                       # (p, 2)
    z = torch.from_numpy(z)                         # (vi, p)
    Ms = torch.from_numpy(Ms)                       # (vi, 3, 4)
    views = torch.from_numpy(views)                 # (vo, 3, 4)

    # camera intrinsics
    fov = math.radians(fov)
    fx = fy = 0.5 * h * math.tan((math.pi - fov) / 2)
    cx, cy = w / 2, h / 2
    K = torch.Tensor([fov, fx, fy, cx, cy])         # (5,)

    # transform points to reference frame
    unprojector = Unprojector()
    view_transformer = ViewTransformer()
    xyz_list = []
    for v in range(n_in_views):
        xyz = unprojector(uv, z[v], K)                          # (p, 3)
        R = torch.matmul(Ms[ctr_idx, :, :3], Ms[v, :, :3].t())  # (3, 3)
        t = Ms[ctr_idx, :, 3:] - torch.matmul(R, Ms[v, :, 3:])  # (3, 1)
        M = torch.cat([R, t], -1)                               # (3, 4)
        xyz = view_transformer(xyz, M)                          # (p, 3)
        xyz_list.append(xyz)
    xyz = torch.cat(xyz_list)
    rgb = rgb.reshape(-1, 3)

    # camera poses for rendering
    for v in range(n_out_views):
        R = torch.matmul(views[v, :, :3], Ms[ctr_idx, :, :3].t())
        t = views[v, :, 3:] - torch.matmul(R, Ms[ctr_idx, :, 3:])
        views[v] = torch.cat([R, t], -1)

    rgb = rgb.t()                                   # (3, p)
    fovs = torch.Tensor([fov] * n_out_views)

    xyz = xyz[None].cuda()                          # (1, p, 3)
    rgb = rgb[None].cuda()                          # (1, 3, p)
    K = K[None].cuda()                              # (1, 5)
    Ms = views[None].cuda()                         # (1, v, 3, 4)
    fovs = fovs[None].cuda()                        # (1, v)

    renderer = Renderer().cuda()

    # render
    t0 = time.time()
    rgb_list = []
    for i in range(n_out_views):
        new_xyz = view_transformer(xyz, Ms[:, i])
        out_dict = renderer(
            xyz=new_xyz, 
            data=rgb, 
            fov=fovs[:, i], 
            h=h // 2 if args.anti_aliasing else h, 
            w=w // 2 if args.anti_aliasing else w, 
            anti_aliasing=args.anti_aliasing,
            denoise=True
        )
        rgb_list.append(out_dict['data'])
    t1 = time.time()
    print('render time: {:s}'.format(time_str(t1 - t0)))

    rgbs = torch.stack(rgb_list, 1)[0]              # (v, 3, h, w)
    rgbs = rgbs.permute(0, 2, 3, 1)                 # (v, h, w, 3)
    rgbs = np.clip(rgbs.cpu().numpy(), 0, 1)
    rgbs = (rgbs * 255).astype(np.uint8)

    # save
    if len(rgbs) > 1:
        imageio.mimwrite(
            os.path.join(save_path, 'video_raw.mp4'), rgbs, fps=30, quality=8
        )
        # for i in range(len(rgbs)):
        #     rgb = Image.fromarray(rgbs[i])
        #     rgb.save(os.path.join(save_path, 'raw_{:03d}.png'.format(i + 1)))
    else:
        rgb = Image.fromarray(rgbs[0])
        rgb.save(os.path.join(save_path, 'out.png'))

###############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='mvs_render', 
                        help='job name')
    parser.add_argument('-g', '--gpu', type=str, default='0', 
                        help='GPU device ID')

    parser.add_argument('-d', '--data_path', type=str, 
                        help='data path')

    parser.add_argument('-aa', '--anti_aliasing', action='store_true',
                        help='if True, apply anti-aliasing')
    
    args = parser.parse_args()

    check_file(args.data_path)

    # set up save folder
    root = 'test/out/mvs_render'
    os.makedirs(root, exist_ok=True)
    save_path = os.path.join(root, args.name)
    ensure_path(save_path)

    set_gpu(args.gpu)

    main(args)