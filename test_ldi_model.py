import os
import math
import time
import random
import argparse

import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import imageio
from PIL import Image

from lib.model import Model3D
from lib.camera import *
from lib.util import *


def main(args):
    # load model
    try:
        ckpt = torch.load(args.model_path)
        config = ckpt['config']
        net = Model3D(config['encoder'], config['decoder'])
        if args.style_path is not None:
            net.convert_for_stylization(config['stylizer'])
        net.cuda()

        net_state = ckpt['netG'] if 'netG' in ckpt.keys() else ckpt['net']
        net.load_state_dict(net_state)
        print('model loaded')
    except:
        raise ValueError(
            '[ERROR] model loading failed: {:s}'.format(args.model_path)
        )

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

    rgb = torch.from_numpy(rgb)                     # (p, 3)
    uv = torch.from_numpy(uv)                       # (p, 2)
    z = torch.from_numpy(z)                         # (p)

    # shuffle points
    ## NOTE: The point cloud encoder gathers a fixed number of points 
    ## from within a neighborhood in a first-come-first-serve manner. 
    ## Shuffling erases any implicit ordering of points in the input LDI. 
    pt_idx = list(range(n_pts))
    random.shuffle(pt_idx)
    rgb, uv, z = rgb[pt_idx], uv[pt_idx], z[pt_idx]
    
    rgb = rgb.transpose(0, 1)                       # (3, p)

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

    input_dict = {
        'src_rgb': rgb[None].cuda(),        # (1, 3, p)
        'src_uv': uv[None].cuda(),          # (1, p, 2)
        'src_z': z[None].cuda(),            # (1, p, 1)
        'K': K[None].cuda(),                # (1, 5)
        'Ms': Ms[None].cuda(),              # (1, v, 3, 4)
        'tgt_fovs': fovs[None].cuda()       # (1, v)
    }

    # load style
    if args.style_path is not None:
        try:
            style = Image.open(args.style_path).convert('RGB')
            style = style.resize((args.style_size, args.style_size))
            style = np.array(style, dtype=np.float32) / 255
            print('style loaded')
        except:
            raise IOError(
                '[ERROR] style loading failed: {:s}'.format(args.style_path)
            )

        style = torch.from_numpy(style).permute(2, 0, 1)    # (3, h, w)
        input_dict['style'] = style[None].cuda()            # (1, 3, h, w)
    
    # re-project and render
    net.eval()
    t0 = time.time()
    with torch.no_grad():
        output_dict = net(
            input_dict=input_dict, 
            h=h // 2,
            w=w // 2,
            pcd_size=args.pcd_size,
            pcd_scale=args.pcd_scale,
            rgb_only=True
        )
    t1 = time.time()
    print('render time: {:s}'.format(time_str(t1 - t0)))
    
    pred_rgbs = output_dict['pred_rgb'][0]                  # (v, 3, h, w)
    tgt_rgbs = output_dict['tgt_rgb'][0]                    # (v, 3, h, w) 
    pred_rgbs = pred_rgbs.permute(0, 2, 3, 1).cpu().numpy() # (v, h, w, 3)
    tgt_rgbs = tgt_rgbs.permute(0, 2, 3, 1).cpu().numpy()   # (v, h, w, 3)
    pred_rgbs = np.clip(pred_rgbs, 0, 1)
    tgt_rgbs = np.clip(tgt_rgbs, 0, 1)
    pred_rgbs = (pred_rgbs * 255).astype(np.uint8)
    tgt_rgbs = (tgt_rgbs * 255).astype(np.uint8)

    # save
    if args.save_frames:
        for i in range(len(pred_rgbs)):
            pred_rgb = Image.fromarray(pred_rgbs[i])
            tgt_rgb = Image.fromarray(tgt_rgbs[i])
            pred_rgb.save(
                os.path.join(save_path, '{:03d}_pred.png'.format(i + 1))
            )
            tgt_rgb.save(
                os.path.join(save_path, '{:03d}_tgt.png'.format(i + 1))
            )
    imageio.mimwrite(
        os.path.join(save_path, 'video_pred.mp4'), pred_rgbs, fps=30, quality=8
    )
    imageio.mimwrite(
        os.path.join(save_path, 'video_tgt.mp4'), tgt_rgbs, fps=30, quality=8
    )

###############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='test', 
                        help='job name')
    parser.add_argument('-g', '--gpu', type=str, default='0', 
                        help='GPU device ID')

    parser.add_argument('-m', '--model_path', type=str,
                        help='model path')
    parser.add_argument('-ldi', '--ldi_path', type=str, 
                        help='LDI path')
    parser.add_argument('-s', '--style_path', type=str, default=None,
                        help='style image path')

    parser.add_argument('-fov', '--fov', type=float, default=None,
                        help='output (vertical) field of view')

    parser.add_argument('-ps', '--pcd_size', type=int, default=None,
                        help='point cloud size')
    parser.add_argument('-pc', '--pcd_scale', type=float, default=1,
                        help='point cloud scale')
    parser.add_argument('-ss', '--style_size', type=int, default=256,
                        help='style image size')
    
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
    os.makedirs('test/out/ldi_model', exist_ok=True)
    save_name = args.name
    if args.style_path is not None:
        style_name = os.path.basename(args.style_path).split('.')[0]
        save_name += '_' + style_name
    save_name += '_' + args.motion
    save_path = os.path.join('test/out/ldi_model', save_name)
    ensure_path(save_path)

    set_gpu(args.gpu)

    main(args)