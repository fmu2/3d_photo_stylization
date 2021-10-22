import os
import glob
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
from lib.render import view2ndc, ndc2view
from lib.camera import *
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
        net.convert_for_stylization(config['stylizer'])
        net.cuda()
        net.load_state_dict(ckpt['net'])
        net.eval()
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
    elif args.motion == 'static':
        Ms = torch.zeros(1, 3, 4)
        Ms[..., :3] += torch.eye(3)

    sio.savemat(
        os.path.join(save_path, 'views.mat'),
        {'Ms': Ms.numpy(), 'fovs': fovs.numpy()}
    )

    rgb = rgb[None].cuda()          # (1, 3, p)
    uv = uv[None].cuda()            # (1, p, 2)
    z = z[None].cuda()              # (1, p, 1)
    K = K[None].cuda()              # (1, 5)
    Ms = Ms[None].cuda()            # (1, v, 3, 4)
    tgt_fovs = fovs[None].cuda()    # (1, v)

    # load style
    style_list = []
    style_paths = glob.glob(os.path.join(args.style_dir, '*.jpg'))
    style_paths = random.sample(
        style_paths, min(len(style_paths), args.n_styles)
    )
    n_styles = len(style_paths)
    for i in range(n_styles):
        try:
            style = Image.open(style_paths[i]).convert('RGB')
            style = style.resize((args.style_size, args.style_size))
            style = np.array(style, dtype=np.float32) / 255
            style_list.append(style)
        except:
            raise IOError(
                '[ERROR] style loading failed: {:s}'.format(style_paths[i])
            )
    print('{:d} styles loaded'.format(n_styles))
    style = np.stack(style_list)                        # (s, h, w, 3)
    style = torch.from_numpy(style).permute(0, 3, 1, 2) # (s, 3, h, w)
    style = style.cuda()

    fov = K[:, 0]               # input FOV
    n_views = Ms.size(1)        # number of views
    h = h // 2 if args.anti_aliasing else h
    w = w // 2 if args.anti_aliasing else w

    # prepare point cloud
    xyz = net.unprojector(uv, z, K)
    raw_xyz, raw_rgb = xyz, rgb

    # view space -> NDC space
    ## NOTE: assume that valid points have depth < 1e5.
    if args.ndc:
        z = xyz[..., 2]
        near = 0.99 * z.amin(1)
        far = (z * (z < 1e5)).quantile(0.95, 1)
        far = torch.maximum(far, near * 2)
        ## NOTE: preserve aspect ratio by setting ar=1
        xyz_ndc = view2ndc(xyz, near, far, fov, ar=1) * args.pcd_scale
    else:
        xyz_ndc = xyz

    # feature extraction
    t0 = time.time()
    with torch.no_grad():
        enc_dict = net.encoder(xyz_ndc, rgb, up=False)
    t1 = time.time()
    print('encode time: {:s}'.format(time_str(t1 - t0)))
    xyz_ndc_list, feats_list = enc_dict['xyz_list'], enc_dict['feats_list']
    feats = feats_list[-1]

    ## NOTE: encoder is only run once.
    ## we then loop over all styles to render stylized videos
    style_time = render_time = 0
    for i in range(n_styles):
        # stylization
        t0 = time.time()
        with torch.no_grad():
            style_dict = net.stylizer(
                style[i:i + 1], xyz_ndc_list, feats_list, 
                up=(net.decoder_up == 1 and net.render_then_decode)
            )
        t1 = time.time()
        style_time += t1 - t0
        feats, up_feats = style_dict['feats'], style_dict.get('up_feats')

        # stylized rendering
        t0 = time.time()
        pred_rgb_list = []
        for k in range(n_views):
            tgt_fov = tgt_fovs[:, k]
            if net.decoder_up > 1:
                xyz_ndc = xyz_ndc_list[-1] / args.pcd_scale
                if args.ndc:
                    xyz = ndc2view(xyz_ndc, near, far, fov, ar=1)
                else:
                    xyz = xyz_ndc
            new_xyz = net.view_transformer(xyz, Ms[:, k])

            if net.render_then_decode:
                # 1) rasterize featurized point cloud to novel view
                # 2) decode 2D feature maps into RGB image
                pred_dict = net.renderer(
                    new_xyz, up_feats if up_feats is not None else feats, 
                    tgt_fov, h // net.decoder_up, w // net.decoder_up, 
                    anti_aliasing=args.anti_aliasing, return_uv=False
                )
                pred_feats = pred_dict['data']
                with torch.no_grad():
                    pred_rgb = net.decoder(pred_feats)
            else:
                # 1) decode featurized point cloud into RGB point cloud
                # 2) rasterize RGB point cloud to novel view
                with torch.no_grad():
                    rgb = net.decoder(xyz_ndc_list, feats)
                pred_dict = net.renderer(
                    new_xyz, rgb, tgt_fov, h, w, 
                    anti_aliasing=args.anti_aliasing, return_uv=False
                )
                pred_rgb = pred_dict['data']

            pred_rgb_list.append(pred_rgb)
        t1 = time.time()
        render_time += t1 - t0

        pred_rgbs = torch.stack(pred_rgb_list, 1)
        pred_rgbs = pred_rgbs[0].permute(0, 2, 3, 1).cpu().numpy()
        pred_rgbs = np.clip(pred_rgbs, 0, 1)
        pred_rgbs = (pred_rgbs * 255).astype(np.uint8)

        # save stylized output
        style_name = os.path.basename(style_paths[i]).split('.')[0]
        if len(pred_rgbs) > 1:
            imageio.mimwrite(
                os.path.join(save_path, 'video_{:s}.mp4'.format(style_name)), 
                pred_rgbs, fps=30, quality=8
            )
        else:
            pred_rgb = Image.fromarray(pred_rgbs[0])
            pred_rgb.save(
                os.path.join(save_path, '{:s}.png'.format(style_name))
            )

    print('stylize time: {:s}'.format(time_str(style_time / n_styles)))
    print('render time: {:s}'.format(time_str(render_time / n_styles)))

    # raw rendering
    tgt_rgb_list = []
    for k in range(n_views):
        tgt_fov = tgt_fovs[:, k]
        new_raw_xyz = net.view_transformer(raw_xyz, Ms[:, k])
        tgt_dict = net.renderer(
            new_raw_xyz, raw_rgb, tgt_fov, h, w, denoise=True
        ) 
        tgt_rgb_list.append(tgt_dict['data'])
    tgt_rgbs = torch.stack(tgt_rgb_list, 1)
    tgt_rgbs = tgt_rgbs[0].permute(0, 2, 3, 1).cpu().numpy()
    tgt_rgbs = np.clip(tgt_rgbs, 0, 1)
    tgt_rgbs = (tgt_rgbs * 255).astype(np.uint8)

    # save raw output
    if len(tgt_rgbs) > 1:
        imageio.mimwrite(
            os.path.join(save_path, 'video_raw.mp4'), 
            tgt_rgbs, fps=30, quality=8
        )
        for i in range(len(tgt_rgbs)):
            tgt_rgb = Image.fromarray(tgt_rgbs[i])
            tgt_rgb.save(
                os.path.join(save_path, 'raw_{:03d}.png'.format(i + 1))
            )
    else:
        tgt_rgb = Image.fromarray(tgt_rgbs[0])
        tgt_rgb.save(os.path.join(save_path, 'raw.png'))

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
    parser.add_argument('-s', '--style_dir', type=str, default=None,
                        help='style image directory')
    parser.add_argument('-ns', '--n_styles', type=int, default=20,
                        help='number of styles')

    parser.add_argument('-fov', '--fov', type=float, default=None,
                        help='output (vertical) field of view')

    parser.add_argument('-ndc', '--ndc', action='store_true',
                        help='if True, convert to NDC space')
    parser.add_argument('-ps', '--pcd_size', type=int, default=None,
                        help='point cloud size')
    parser.add_argument('-pc', '--pcd_scale', type=float, default=1,
                        help='point cloud scale')
    parser.add_argument('-ss', '--style_size', type=int, default=256,
                        help='style image size')
    
    parser.add_argument('-cam', '--motion', type=str, default='zoom', 
                        choices=('zoom', 'dolly_zoom', 'ken_burns', 
                                 'swing', 'circle', 'static'
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
    parser.add_argument('-aa', '--anti_aliasing', action='store_true',
                        help='if True, apply anti-aliasing')
    
    args = parser.parse_args()

    check_file(args.ldi_path)

    # set up save folder
    os.makedirs('test/out/batch_stylize', exist_ok=True)
    save_name = args.name + '_' + args.motion
    save_path = os.path.join('test/out/batch_stylize', save_name)
    ensure_path(save_path)

    set_gpu(args.gpu)

    main(args)