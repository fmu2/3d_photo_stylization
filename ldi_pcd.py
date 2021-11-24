import os
import math
import random
import argparse

import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import open3d as o3d
from plyfile import PlyData, PlyElement

from lib.module import Unprojector
from lib.render import view2ndc
from lib.util import *


def int16_to_hex_str(color):
    hex_str = ""
    color_map = {i: str(i) for i in range(10)}
    color_map.update({10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F"})
    hex_str += color_map[color // 16]
    hex_str += color_map[color % 16]
    return hex_str


def rgb_to_hex_str(*rgb):
    hex_str = "#"
    for item in rgb:
        hex_str += int16_to_hex_str(item)
    return hex_str


def rgbs_to_hex_strs(rgb_list):
    hex_list = []
    for rgb in rgb_list:
        hex_list.append(rgb_to_hex_str(rgb[0], rgb[1], rgb[2]))
    return hex_list


def plot_point_cloud(xyz, rgb, path):
    """
    Display colorized 3D point cloud.

    Args:
        xyz (float array, (p, 3)): xyz coordinates.
        rgb (float array, (p, 3)): rgb color values.
    """
    rgb = rgbs_to_hex_strs(rgb)
    X, Y, Z = xyz[:, 0], xyz[:, 2], xyz[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, -Z, c=rgb, s=5, depthshade=True)
    # plt.show()
    plt.savefig(os.path.join(path, 'pcd.png'), transparent=True, dpi=300)
    plt.close(fig)


def save_point_cloud(xyz, rgb, path):
    """
    Save colorized 3D point cloud as a PLY file.

    Args:
        xyz (float array, (p, 3)): xyz coordinates.
        rgb (float array, (p, 3)): rgb color values.
    """
    pcd = np.hstack([xyz, rgb])
    python_types = (float, float, float, int, int, int)
    np_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
               ]

    vertices = []
    for i in range(len(pcd)):
        vertices.append(tuple(dtype(v) for dtype, v in zip(python_types, pcd[i])))
    vertices_array = np.array(vertices, dtype=np_types)
    el = PlyElement.describe(vertices_array, 'vertex')

    pcd_path = os.path.join(path, 'pcd.ply')

    # save
    PlyData([el]).write(pcd_path)

    # visualize
    pcd = o3d.io.read_point_cloud(pcd_path)
    o3d.visualization.draw_geometries([pcd])


def main(args):
    # load LDI
    try:
        ldi = sio.loadmat(args.ldi_path)
        fov, h, w = float(ldi['fov']), int(ldi['h']), int(ldi['w'])
        rgb = ldi['rgb'].astype(np.uint8)
        uv = ldi['uv'].astype(np.float32) + 0.5
        z = ldi['z'].astype(np.float32)
        n_pts = len(rgb)
        print('LDI loaded')
    except:
        raise IOError(
            '[ERROR] LDI loading failed: {:s}'.format(args.ldi_path)
        )
    uv = torch.from_numpy(uv)[None]             # (1, p, 2)
    z = torch.from_numpy(z)[None]               # (1, p)

    # camera intrinsics
    fov = math.radians(fov)
    fx = fy = 0.5 * h * math.tan((math.pi - fov) / 2)
    cx, cy = w / 2, h / 2
    K = torch.Tensor([[fov, fx, fy, cx, cy]])   # (1, 5)

    # construct point cloud
    unprojector = Unprojector()
    xyz = unprojector(uv, z, K)
    if args.ndc:
        z = xyz[..., 2]
        near = 0.99 * z.amin(1)
        far = (z * (z < 1e5)).quantile(0.95, 1)
        far = torch.maximum(far, near * 2)
        xyz = view2ndc(xyz, near, far, K[:, 0])
    xyz = xyz[0].numpy()
    save_point_cloud(xyz, rgb, save_path)

    ###########################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='ldi',
                        help='job name')
    parser.add_argument('-ldi', '--ldi_path', type=str, 
                        help='image path')
    parser.add_argument('-ndc', '--ndc', action='store_true',
                        help='if True, save point cloud in NDC space')
    
    args = parser.parse_args()

    check_file(args.ldi_path)

    # set up save folder
    os.makedirs('test/out/ldi_pcd', exist_ok=True)
    save_path = os.path.join('test/out/ldi_pcd', args.name)
    ensure_path(save_path)

    main(args)