import os
import math
import random

import numpy as np
import scipy.io as sio
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms


###############################################################################
""" Style """

class WikiArtDataset(Dataset):
    
    def __init__(self, root, split, im_size=256):
        super(WikiArtDataset, self).__init__()

        self.root = root
        self.im_size = im_size

        assert split in ('train', 'val'), \
            '[ERROR] invalid split: {:s}'.format(split)
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(im_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(im_size),
                transforms.CenterCrop(im_size),
                transforms.ToTensor()
            ])

        self.data_list = os.listdir(os.path.join(root, split))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.split, self.data_list[idx])
        try:
            im = Image.open(path).convert('RGB')
        except:
            raise IOError('[ERROR] image loading failed: {:s}'.format(path))
        im = self.transform(im)
        return im

###############################################################################
""" Content """

class COCODataset(Dataset):

    def __init__(self, root, split, im_size=256):
        super(COCODataset, self).__init__()

        self.root = root
        self.im_size = im_size

        assert split in ('train', 'val'), \
            '[ERROR] invalid split: {:s}'.format(split)
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(im_size),
                transforms.RandomCrop(im_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(im_size),
                transforms.CenterCrop(im_size),
                transforms.ToTensor()
            ])

        self.data_list = os.listdir(os.path.join(root, split))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.split, self.data_list[idx])
        try:
            im = Image.open(path).convert('RGB')
        except:
            raise IOError('[ERROR] image loading failed: {:s}'.format(path))
        im = self.transform(im)

        lib = {'src_rgb': im}
        return lib


class PointCloudDataset(Dataset):

    def __init__(
        self, root, split, n_target_views=1,
        x_lim=[-0.02, 0.02], y_lim=[-0.02, 0.02], z_lim=[-0.05, 0.15]
    ):
        """
        Args:
            root (str): data directory.
            split (str): split name ('train', 'val').
            n_target_views (int): number of target views.
            x_lim (float list/tuple): left / right bounds.
            y_lim (float list/tuple): top / bottom bounds.
            z_lim (float list/tuple): near / far bounds.
        """
        super(PointCloudDataset, self).__init__()

        self.root = root

        assert split in ('train', 'val'), \
            '[ERROR] invalid split: {:s}'.format(split)
        self.split = split

        self.data_path = os.path.join(root, split, 'ldi', 'ldi')
        self.data_list = os.listdir(self.data_path)

        assert n_target_views in (1, 2, 3, 4), \
            ('[ERROR] number of target views must be 1, 2, 3 or 4, '
             'got {:d}'.format(n_target_views)
            )
        self.n_target_views = n_target_views

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
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.z_lim = z_lim

        # extreme views
        R = np.eye(3)[None].repeat(8, 0)
        ts = -np.array(
            [[x_lim[0], y_lim[0], z_lim[0]],
             [x_lim[1], y_lim[0], z_lim[0]],
             [x_lim[0], y_lim[1], z_lim[0]],
             [x_lim[1], y_lim[1], z_lim[0]],
             [x_lim[0], y_lim[0], z_lim[1]],
             [x_lim[1], y_lim[0], z_lim[1]],
             [x_lim[0], y_lim[1], z_lim[1]],
             [x_lim[1], y_lim[1], z_lim[1]]]
        )[..., None]
        extreme_Ms = np.concatenate([R, ts], -1)
        extreme_Ms = torch.from_numpy(extreme_Ms.astype(np.float32))
        self.extreme_Ms = extreme_Ms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path = os.path.join(self.data_path, self.data_list[idx])
        try:
            ldi = sio.loadmat(path)
        except:
            raise IOError('[ERROR] LDI loading failed: {:s}'.format(path))
        
        rgb = torch.from_numpy(ldi['rgb'].astype(np.float32) / 255) # (p, 3)
        uv = torch.from_numpy(ldi['uv'].astype(np.float32) + 0.5)   # (p, 2)
        z = torch.from_numpy(ldi['z'].astype(np.float32))           # (p, 1)
        n_pts = torch.tensor(len(rgb), dtype=torch.int)
        pt_idx = list(range(len(rgb)))
        random.shuffle(pt_idx)  # shuffle points
        rgb, uv, z = rgb[pt_idx], uv[pt_idx], z[pt_idx]
        rgb = rgb.transpose(1, 0)                                   # (3, p)

        # intrinsics
        fov, h, w = float(ldi['fov']), int(ldi['h']), int(ldi['w'])
        src_fov = math.radians(fov)
        fx = fy = 0.5 * h * math.tan((math.pi - src_fov) / 2)
        cx, cy = w / 2, h / 2
        K = torch.Tensor([src_fov, fx, fy, cx, cy])
        tgt_fovs = torch.Tensor(
            [math.radians(random.uniform(0.95, 1.05) * fov) \
                for _ in range(self.n_target_views)
            ]
        )

        # extrinsics
        R = np.eye(3)[None].repeat(self.n_target_views, 0)          # [v, 3, 3]
        xs = np.random.uniform(*self.x_lim, self.n_target_views)    # [v,]
        ys = np.random.uniform(*self.y_lim, self.n_target_views)    # [v,]
        zs = np.random.uniform(*self.z_lim, self.n_target_views)    # [v,]
        ts = np.stack([-xs, -ys, -zs], -1)[..., None]               # [v, 3, 1]
        Ms = np.concatenate([R, ts], -1)                            # [v, 3, 4]
        Ms = torch.from_numpy(Ms.astype(np.float32))

        lib = {
            'n_pts': n_pts,
            'src_rgb': rgb,
            'src_uv': uv,
            'src_z': z,
            'K': K,
            'Ms': Ms,
            'extreme_Ms': self.extreme_Ms,
            'tgt_fovs': tgt_fovs,
        }
        return lib


def make_dataset(config, root, split='train', nvs=False):
    if config['name'] == 'pcd':
        assert nvs, '[ERROR] PointCloud only supports novel view synthesis'
        dataset = PointCloudDataset(
            root, split, 
            n_target_views=config.get('n_target_views', 1),
            x_lim=config.get('x_lim', [-0.02, 0.02]),
            y_lim=config.get('y_lim', [-0.02, 0.02]),
            z_lim=config.get('z_lim', [-0.05, 0.15])
        )
    elif config['name'] == 'coco':
        assert not nvs, '[ERROR] COCO does not support novel view synthesis'
        dataset = COCODataset(root, split, im_size=config['im_size'])
    elif config['name'] == 'wikiart':
        assert not nvs, '[ERROR] WikiArt does not support novel view synthesis'
        dataset = WikiArtDataset(root, split, im_size=config['im_size'])
    else:
        raise NotImplementedError(
            '[ERROR] invalid dataset: {:s}'.format(config['name'])
        )
    return dataset


def pad_to_same_size(tensor_list, dim=0):
    """
    Pad variable-sized tensors to the same size.
    NOTE: only support padding along a single dimension.

    Args:
        tensor_list (tensor list): a list of tensors.
        dim (int): dimension along which to pad.

    Returns:
        tensor_list (tensor list): a list of tensors with same size.
    """
    max_len = max([t.size(dim) for t in tensor_list])
    for i in range(len(tensor_list)):
        pad = [0] * tensor_list[i].dim() * 2
        pad[-2 * dim - 1] = max_len - tensor_list[i].size(dim)
        tensor_list[i] = F.pad(tensor_list[i], pad, value=1e5)
    return tensor_list


def collate_fn(batch):
    keys = batch[0].keys()
    out = {k: [] for k in keys}
    for k in keys:
        for lib in batch:
            out[k].append(lib[k])
        if 'n_pts' in keys:
            if k == 'src_rgb':
                out['src_rgb'] = torch.stack(
                    pad_to_same_size(out['src_rgb'], 1)
                )   # (bs, 3, p)
            elif k == 'src_z':
                out['src_z'] = torch.stack(
                    pad_to_same_size(out['src_z'], 0)
                )   # (bs, p, 1)
            elif k == 'src_uv':
                out['src_uv'] = torch.stack(
                    pad_to_same_size(out['src_uv'], 0)
                )   # (bs, p, 2)
            else:
                out[k] = torch.stack(out[k])
        else:
            out[k] = torch.stack(out[k])
    return out


def cycle(iterable):
    while True:
        for x in iterable:
            yield x