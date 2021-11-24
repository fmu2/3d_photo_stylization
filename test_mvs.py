import os
import math
import time
import random
import argparse

import torch
import numpy as np
import scipy.io as sio
import imageio
from PIL import Image

from lib.module import Unprojector, ViewTransformer
from lib.model import Model3D
from lib.util import *


def main(args):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load model
    try:
        ckpt = torch.load(args.model_path)
        config = ckpt['config']
        net = Model3D(config['encoder'], config['decoder'])
        if args.style_path is not None:
            net.convert_for_stylization(config['stylizer'])
        net.cuda()

        net_state = ckpt['netG'] if 'netG' in ckpt.keys() else ckpt['net']
        net.load_state_dict(net_state, strict=False)
        net.eval()
        print('model loaded')
    except:
        raise ValueError(
            '[ERROR] model loading failed: {:s}'.format(args.model_path)
        )

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

    # shuffle points
    ## NOTE: The point cloud encoder gathers a fixed number of points 
    ## from within a neighborhood in a first-come-first-serve manner. 
    ## Shuffling erases any implicit ordering of points in the input data. 
    pt_idx = list(range(len(xyz)))
    random.shuffle(pt_idx)
    rgb, xyz = rgb[pt_idx], xyz[pt_idx]
    
    rgb = rgb.t()                           # (3, p)
    fovs = torch.Tensor([fov] * n_out_views)

    input_dict = {
        'src_rgb': rgb[None].cuda(),        # (1, 3, p)
        'src_xyz': xyz[None].cuda(),        # (1, p, 3)
        'K': K[None].cuda(),                # (1, 5)
        'Ms': views[None].cuda(),           # (1, vo, 3, 4)
        'tgt_fovs': fovs[None].cuda()       # (1, vo)
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
    t0 = time.time()
    with torch.no_grad():
        output_dict = net(
            input_dict=input_dict, 
            h=h // 2 if args.anti_aliasing else h,
            w=w // 2 if args.anti_aliasing else w,
            ndc=args.ndc,
            pcd_size=args.pcd_size,
            pcd_scale=args.pcd_scale,
            anti_aliasing=args.anti_aliasing,
            rgb_only=True
        )
    t1 = time.time()
    print('render time: {:s}'.format(time_str(t1 - t0)))
    
    pred_rgbs = output_dict['pred_rgb'][0]                  # (vo, 3, h, w)
    tgt_rgbs = output_dict['tgt_rgb'][0]                    # (vo, 3, h, w)
    pred_rgbs = pred_rgbs.permute(0, 2, 3, 1).cpu().numpy() # (vo, h, w, 3)
    tgt_rgbs = tgt_rgbs.permute(0, 2, 3, 1).cpu().numpy()   # (vo, h, w, 3)
    pred_rgbs = np.clip(pred_rgbs, 0, 1)
    tgt_rgbs = np.clip(tgt_rgbs, 0, 1)
    pred_rgbs = (pred_rgbs * 255).astype(np.uint8)
    tgt_rgbs = (tgt_rgbs * 255).astype(np.uint8)

    # save
    imageio.mimwrite(
        os.path.join(save_path, 'video_pred.mp4'), pred_rgbs, fps=30, quality=8
    )
    imageio.mimwrite(
        os.path.join(save_path, 'video.mp4'), tgt_rgbs, fps=30, quality=8
    )

###############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='test_mvs', 
                        help='job name')
    parser.add_argument('-g', '--gpu', type=str, default='0', 
                        help='GPU device ID')

    parser.add_argument('-m', '--model_path', type=str,
                        help='model path')
    parser.add_argument('-d', '--data_path', type=str, 
                        help='data path')
    parser.add_argument('-s', '--style_path', type=str, default=None,
                        help='style image path')

    parser.add_argument('-ndc', '--ndc', action='store_true',
                        help='if True, convert to NDC space')
    parser.add_argument('-ps', '--pcd_size', type=int, default=None,
                        help='point cloud size')
    parser.add_argument('-pc', '--pcd_scale', type=float, default=1,
                        help='point cloud scale')
    parser.add_argument('-ss', '--style_size', type=int, default=256,
                        help='style image size')

    parser.add_argument('-aa', '--anti_aliasing', action='store_true',
                        help='if True, apply anti-aliasing')
    
    args = parser.parse_args()

    check_file(args.data_path)

    # set up save folder
    root = 'test/out/mvs_model'
    os.makedirs(root, exist_ok=True)
    save_name = args.name
    if args.style_path is not None:
        style_name = os.path.basename(args.style_path).split('.')[0]
        save_name += '_' + style_name
    save_path = os.path.join(root, save_name)
    ensure_path(save_path)

    set_gpu(args.gpu)

    main(args)