import os
import math
import time
import argparse

import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import imageio
from PIL import Image

from lib.module import Unprojector, ViewTransformer, Renderer
from lib.camera import *
from lib.util import *


def main(args):
    # load LDI
    try:
        ldi = sio.loadmat(args.ldi_path)
        fov, h, w = float(ldi['fov']), int(ldi['h']), int(ldi['w'])
        rgb = ldi['rgb'].astype(np.float32) / 255
        uv = ldi['uv'].astype(np.float32) + 0.5
        z = ldi['z'].astype(np.float32)
        n_pts = len(rgb)
        print('LDI loaded')
    except:
        raise IOError(
            '[ERROR] LDI loading failed: {:s}'.format(args.ldi_path)
        )

    rgb = torch.from_numpy(rgb).transpose(1, 0)     # (3, p)
    uv = torch.from_numpy(uv)                       # (p, 2)
    z = torch.from_numpy(z)                         # (p,)

    # camera intrinsics
    fov = math.radians(fov)
    fx = fy = 0.5 * h * math.tan((math.pi - fov) / 2)
    cx, cy = w / 2, h / 2
    K = torch.Tensor([fov, fx, fy, cx, cy])         # (5,)

    # camera poses
    out_fov = math.radians(args.fov) if args.fov is not None else fov
    fovs = torch.Tensor([out_fov] * args.n_frames)
    if args.motion == 'zoom':
        Ms = make_zoom(args.n_frames, args.z_lim)
    elif args.motion == 'dolly_zoom':
        ctr_idx = torch.logical_and(
            torch.logical_and(uv[:, 0] / w > 0.49, uv[:, 0] / w < 0.51), 
            torch.logical_and(uv[:, 1] / h > 0.49, uv[:, 1] / h < 0.51)
        )
        ctr_depth = z[ctr_idx].min()
        Ms, fovs = make_dolly_zoom(args.n_frames, args.z_lim, out_fov, ctr_depth)
    elif args.motion == 'ken_burns':
        Ms = make_ken_burns(args.n_frames, args.x_lim, args.y_lim, args.z_lim)
    elif args.motion == 'swing':
        Ms = make_swing(args.n_frames, args.x_lim, args.z_lim)
    elif args.motion == 'circle':
        Ms = make_circle(args.n_frames, args.x_lim, args.y_lim, args.z_lim)

    uv = uv[None].cuda()                            # (1, p, 2)
    z = z[None].cuda()                              # (1, p)
    rgb = rgb[None].cuda()                          # (1, 3, p)
    K = K[None].cuda()                              # (1, 5)
    Ms = Ms[None].cuda()                            # (1, v, 3, 4)
    fovs = fovs[None].cuda()                        # (1, v)

    # set up rendering utilities
    unprojector = Unprojector().cuda()
    view_transformer = ViewTransformer().cuda()
    renderer = Renderer(anti_aliasing=args.anti_aliasing).cuda()
    
    # re-project and render
    t0 = time.time()
    xyz = unprojector(uv, z, K)
    rgb_list = []
    for i in range(args.n_frames):
        new_xyz = view_transformer(xyz, Ms[:, i])
        out_dict = renderer(
            xyz=new_xyz, 
            data=rgb, 
            fov=fovs[:, i], 
            h=h // args.anti_aliasing, 
            w=w // args.anti_aliasing, 
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
    if args.save_frames:
        for i in range(len(rgbs)):
            rgb = Image.fromarray(rgbs[i])
            rgb.save(os.path.join(save_path, '{:03d}.png'.format(i + 1)))
    imageio.mimwrite(
        os.path.join(save_path, 'video.mp4'), rgbs, fps=30, quality=8
    ) 

    ###########################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='render', 
                        help='job name')
    parser.add_argument('-g', '--gpu', type=str, default='0', 
                        help='GPU device ID')

    parser.add_argument('-ldi', '--ldi_path', type=str, 
                        help='LDI path')

    parser.add_argument('-aa', '--anti_aliasing', type=int, default=1,
                        help='supersampling rate for anti-aliasing')
    parser.add_argument('-fov', '--fov', type=float, default=None,
                        help='output (vertical) field of view')
    
    parser.add_argument('-cam', '--motion', type=str, default='zoom', 
                        choices=('zoom', 'dolly_zoom', 'ken_burns', 
                                 'swing', 'circle',
                                ),
                        help='camera motion')
    
    parser.add_argument('-x', '--x_lim', type=float, nargs='+', 
                        default=[-0.05, 0.05], help='left / right bounds')
    parser.add_argument('-y', '--y_lim', type=float, nargs='+',
                        default=[-0.05, 0.05], help='top / bottom bounds')
    parser.add_argument('-z', '--z_lim', type=float, nargs='+',
                        default=[0, 0.15], help='near / far bounds')
    
    parser.add_argument('-f', '--n_frames', type=int, 
                        default=90, help='number of frames')
    parser.add_argument('-sf', '--save_frames', action='store_true', 
                        default=False, help='if True, save all frames')
    
    args = parser.parse_args()

    check_file(args.ldi_path)

    # set up save folder
    os.makedirs('test/out/ldi_render', exist_ok=True)
    save_name = args.name + '_' + args.motion
    save_path = os.path.join('test/out/ldi_render', save_name)
    ensure_path(save_path)

    set_gpu(args.gpu)

    main(args)