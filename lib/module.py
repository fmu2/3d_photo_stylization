import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg import NormalizedVGG, make_dvgg
from .render import view2ndc, Splatter
from .pointnet2 import PointUpsample, PointDownsample
from .filter import PartialMeanFilter
from .layer import *

###############################################################################
""" Differentiable point cloud renderer """

class Unprojector(nn.Module):
    """ Unproject image pixels to 3D point cloud. """

    def __init__(self):
        super(Unprojector, self).__init__()

    def forward(self, uv, z, K):
        """
        Args:
            uv (float tensor, (bs, p, 2)): uv-coordinates of points.
            z (float tensor, (bs, p, (1))): view-space depth of points.
            K (float tensor, (bs, 5)): camera intrinsics (fov, fx, fy, cx, cy).

        Returns:
            xyz (float tensor, (bs, p, 3)): view-space xyz-coordinates of points.
        """
        fx, fy = K[..., 1:2], K[..., 2:3]   # focal length
        cx, cy = K[..., 3:4], K[..., 4:5]   # principal point
        if z.dim() == 3:
            z = z[..., 0]
        x = (uv[..., 0] - cx) * z / fx
        y = (uv[..., 1] - cy) * z / fy
        xyz = torch.stack([x, y, z], -1)                # (bs, p, 3)
        return xyz


class ViewTransformer(nn.Module):
    """ Transform 3D point coordinates given camera pose. """

    def __init__(self):
        super(ViewTransformer, self).__init__()

    def forward(self, xyz, M, inverse=False):
        """
        Args:
            xyz (float tensor, (bs, p, 3)): point coordinates in view space.
            M (float tensor, (bs, 3, 4)): camera pose.

        Returns:
            xyz (float tensor, (bs, p, 3)): transformed point coordinates.
        """
        M = M.transpose(-1, -2)
        R, t = M[..., :3, :], M[..., 3:, :]             # (bs, 3/1, 3)
        if inverse:
            R = R.transpose(-1, -2)
            xyz = torch.matmul(xyz - t, R)              # (bs, 3, p)
        else:
            xyz = torch.matmul(xyz, R) + t              # (bs, 3, p)
        return xyz


class Renderer(nn.Module):
    """ Rasterize 3D point cloud given camera intrinsics. """

    def __init__(self, anti_aliasing=2):
        super(Renderer, self).__init__()

        self.splatter = Splatter()
        self.anti_aliasing = anti_aliasing
        
        if anti_aliasing > 1:
            self.pool = PartialAvgPool2d(anti_aliasing)
        else:
            self.pool = lambda x, mask: (x, mask)

        self.denoiser = PartialMeanFilter(3)

    def forward(self, xyz, data, fov, h, w, anti_aliasing=True, 
                denoise=False, return_uv=False):
        """
        Args:
            xyz (float tensor, (bs, p, 3)): point coordinates in view space.
            data (float tensor, (bs, c, p)): point-associated data.
            fov (float tensor, (bs,)): vertical angular field of view (unit: rad).
            h (int): height of rendered data maps (unit: px).
            w (int): width of rendered data maps (unit: px).
            anti_aliasing (bool): if True, apply anti-aliasing.
            denoise (bool): if True, apply denoising filter.
            return_uv (bool): if True, return visibility of points and 
                their uv-coordinates following rasterization.

        Returns:
            out_dict (dict):
                data (float tensor, (bs, c, h, w)): rendered data maps.
                mask (bool tensor, (bs, 1, h, w)): mask of valid pixels.
                uv ((optional) float tensor, (bs, p, 2)): uv-coordinates of points.
                viz ((optional) bool tensor, (bs, p)): visibility mask.
        """
        # view space -> NDC space
        ## NOTE: assume that valid points have depth < 1e5.
        z = xyz[..., 2]
        near = 0.99 * z.amin(1)
        far = (z * (z < 1e5)).quantile(0.95, 1)
        far = torch.maximum(far, near * 2)
        xyz = view2ndc(xyz, near, far, fov, w / h)

        # rasterization followed by anti-aliasing
        h = h * self.anti_aliasing if anti_aliasing else h 
        w = w * self.anti_aliasing if anti_aliasing else w
        data, conf, viz = self.splatter(xyz, data, h, w)
        mask = conf > 0
        if denoise:
            data, mask = self.denoiser(data, mask)
        if anti_aliasing:
            data, mask = self.pool(data, mask)

        out_dict = {'data': data, 'mask': mask}
        if return_uv:
            uv = xyz[..., :2]
            out_dict['uv'] = uv
            out_dict['viz'] = viz
        return out_dict

###############################################################################
""" Encoder """

class VGGEncoder(nn.Module):

    def __init__(self, cfg):
        super(VGGEncoder, self).__init__()

        self.layer = cfg.get('layer', 3)
        self.vgg = NormalizedVGG(pool=cfg.get('pool', 'max'))

    def forward(self, rgb, return_pyramid=False):
        feats = self.vgg(rgb)
        if return_pyramid:
            return feats
        return feats[self.layer - 1]


class PointCloudEncoder(nn.Module):

    def __init__(self, cfg):
        super(PointCloudEncoder, self).__init__()

        self.n_levels = cfg['n_levels']
        assert len(cfg['scale_factor']) == self.n_levels
        assert len(cfg['depth']) == self.n_levels
        assert len(cfg['dims']) == self.n_levels
        assert len(cfg['k']) == self.n_levels
        assert len(cfg['radius']) == self.n_levels

        self.in_conv = GCNLayer(
            cfg['in_conv'],
            in_dim=3, 
            out_dim=cfg['in_dim'], 
            scale_factor=1,
            radius=cfg['in_radius'], 
            k=cfg['in_k'], 
            aggregate=cfg.get('aggregate', 'max'),
            norm=cfg.get('norm'), 
            actv=cfg.get('actv', 'relu'),
            res=False
        )

        self.blocks = nn.ModuleList()
        in_dim = cfg['in_dim']

        for i in range(self.n_levels):
            blocks = nn.ModuleList()
            scale_factor = cfg['scale_factor'][i]
            out_dim = cfg['dims'][i]
            for j in range(cfg['depth'][i]):
                blocks.append(
                    GCNBlock(
                        cfg['block'],
                        in_dim=in_dim, 
                        out_dim=out_dim,
                        scale_factor=scale_factor,
                        radius=cfg['radius'][i][j],
                        k=cfg['k'][i][j],
                        aggregate=cfg.get('aggregate', 'max'),
                        norm=cfg.get('norm'), 
                        actv=cfg.get('actv', 'relu'),
                        res=cfg.get('res')
                    )
                )
                scale_factor = 1
                in_dim = out_dim
            self.blocks.append(blocks)

        self.out_fc = nn.Conv1d(in_dim, in_dim, 1)
        self.out_relu = cfg.get('out_relu')

        self.up = PointUpsample(mode=cfg.get('up', 'linear'))

    def _build_pyramid(self, xyz, feats):
        xyz_list, feats_list = [xyz], [feats]
        for i in range(self.n_levels):
            blocks = self.blocks[i]
            for j in range(len(blocks)):
                xyz, feats = blocks[j](xyz, feats)
            xyz_list.append(xyz)
            feats_list.append(feats)
        feats_list[-1] = self.out_fc(feats_list[-1])
        if self.out_relu:
            feats_list[-1] = F.relu(feats_list[-1], inplace=True)
        return xyz_list, feats_list

    def forward(self, xyz, rgb, up=True):
        """
        Args:
            xyz (float tensor, (bs, p, 3)): point coordinates.
            rgb (float tensor, (bs, 3, p)): point RGB values.
            up (bool): if True, upsample output features to input resolution.
        
        Returns:
            output_dict (dict):
                xyz_list (float tensor list): point NDC coordinates at all levels.
            feats (float tensor, (bs, c, p)): output features.
        """
        assert xyz.size(1) == rgb.size(2), \
            ('[ERROR] point cloud size ({:d}) and number of RGB values ({:d}) '
             'must match'.format(xyz.size(1), rgb.size(2))
            )
        output_dict = dict()

        _, feats = self.in_conv(xyz, rgb)
        xyz_list, feats_list = self._build_pyramid(xyz, feats)
        output_dict['xyz_list'] = xyz_list
        output_dict['feats_list'] = feats_list

        xyz, feats = xyz_list[-1], feats_list[-1]
        if up:  # upsample deepest features to input resolution
            for i in range(len(xyz_list) - 2, -1, -1):
                parent_xyz = xyz_list[i]
                feats = self.up(xyz, parent_xyz, feats)
                xyz = parent_xyz
            output_dict['up_feats'] = feats
        return output_dict

###############################################################################
""" Decoder """

class VGGDecoder(nn.Module):

    def __init__(self, cfg):
        super(VGGDecoder, self).__init__()

        self.dvgg = make_dvgg(
            cfg['layer'], cfg['in_dim'], 
            up=cfg.get('up', 'nearest'),
            pretrained=cfg.get('pretrained', False)
        )

    def forward(self, feats):
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        rgb = self.dvgg(feats)
        return rgb


class VGGAttNDecoder(nn.Module):
    """ VGG-style decoder for SANet (Park et al., CVPR 19) and AdaAttN 
    (Liu et al., ICCV 21) """

    def __init__(self, cfg):
        super(VGGAttNDecoder, self).__init__()

        self.layers = cfg['layers']

        up = cfg.get('up', 'nearest')
        if up == 'nearest':
            Upsampling2d = nn.UpsamplingNearest2d
        elif up == 'bilinear':
            Upsampling2d = nn.UpsamplingBilinear2d
        else:
            raise NotImplementedError(
                '[ERROR] invalid upsampling operator: {:s}'.format(up)
            )
        self.up = up

        # input: relu4_1 + relu5_1
        if 5 in self.layers:
            self.in_layer = nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(1), nn.Conv2d(512, 256, 3),
                nn.ReLU(inplace=True),
                Upsampling2d(scale_factor=2),
            )
        else:
            self.in_layer = nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3),
                nn.ReLU(inplace=True),
                Upsampling2d(scale_factor=2),
            )

        self.stage3 = nn.Sequential(
            nn.ReflectionPad2d(1), 
            nn.Conv2d(512 if 3 in self.layers else 256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(256, 128, 3),
            nn.ReLU(inplace=True),
            Upsampling2d(scale_factor=2),
        )

        self.stage2 = nn.Sequential(
            nn.ReflectionPad2d(1), 
            nn.Conv2d(256 if 2 in self.layers else 128, 128, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            Upsampling2d(scale_factor=2),
        )

        self.stage1 = nn.Sequential(
            nn.ReflectionPad2d(1), 
            nn.Conv2d(128 if 1 in self.layers else 64, 64, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 3, 3),
        )

    def _pad(self, x, y):
        dh = y.size(-2) - x.size(-2)
        dw = y.size(-1) - x.size(-1)
        if dh == 0 and dw == 0:
            return x
        if dh < 0:
            x = x[..., :dh, :]
        if dw < 0:
            x = x[..., :, :dw]
        if dh > 0 or dw > 0:
            x = F.pad(
                x, 
                pad=(dw // 2, dw - dw // 2, dh // 2, dh - dh // 2), 
                mode='reflect'
            )
        return x

    def forward(self, feats_list):
        assert len(feats_list) == 5

        x = feats_list[-2]
        if 5 in self.layers:
            x5 = feats_list[-1]
            x5 = F.interpolate(x5, size=x.shape[-2:], mode=self.up)
            x = x + x5
        x = self.in_layer(x)
        
        if 3 in self.layers:
            x3 = feats_list[-3]
            x = torch.cat([self._pad(x, x3), x3], 1)
        x = self.stage3(x)
        
        if 2 in self.layers:
            x2 = feats_list[-4]
            x = torch.cat([self._pad(x, x2), x2], 1)
        x = self.stage2(x)
        
        if 1 in self.layers:
            x1 = feats_list[-5]
            x = torch.cat([self._pad(x, x1), x1], 1)
        rgb = self.stage1(x)
        return rgb


class DownBlock(nn.Module):

    def __init__(self, in_dim, out_dim, down='mean',
                 norm=None, actv='leaky_relu'):
        super(DownBlock, self).__init__()

        if down == 'conv':
            self.down_conv = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 3, 2, 1),
                get_norm(out_dim, norm), get_actv(actv),
                nn.Conv2d(out_dim, out_dim, 3, 1, 1),
                get_norm(out_dim, norm), get_actv(actv),
            )
        elif down == 'mean':
            self.down_conv = nn.AvgPool2d(2)
        else:
            raise NotImplementedError(
                '[ERROR] invalid downsampling operator: {:s}'.format(down)
            )

    def forward(self, x):
        x = self.down_conv(x)
        return x


class UpBlock(nn.Module):

    def __init__(self, in_dim, out_dim, skip_dim=None, up='conv', 
                 norm=None, actv='relu'):
        super(UpBlock, self).__init__()

        if up == 'conv':
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 3, 2, 1, 1),
                get_norm(out_dim, norm), get_actv(actv),
            )
        else:
            assert up in ('bilinear', 'nearest'), \
                '[ERROR] invalid upsampling mode: {:s}'.format(up)
            self.up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=up),
                nn.Conv2d(in_dim, out_dim, 3, 1, 1),
                get_norm(out_dim, norm), get_actv(actv),
            )
        
        in_dim = out_dim
        if skip_dim is not None:
            in_dim += skip_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            get_norm(out_dim, norm), get_actv(actv),
        )

    def _pad(self, x, y):
        dh = y.size(-2) - x.size(-2)
        dw = y.size(-1) - x.size(-1)
        if dh == 0 and dw == 0:
            return x
        if dh < 0:
            x = x[..., :dh, :]
        if dw < 0:
            x = x[..., :, :dw]
        if dh > 0 or dw > 0:
            x = F.pad(
                x, 
                pad=(dw // 2, dw - dw // 2, dh // 2, dh - dh // 2), 
                mode='reflect'
            )
        return x

    def forward(self, x, skip=None):
        x = self.up_conv(x)
        if skip is not None:
            x = torch.cat([self._pad(x, skip), skip], 1)
        x = self.conv(x)
        return x


class UpDecoder(nn.Module):

    def __init__(self, cfg):
        super(UpDecoder, self).__init__()

        in_dim = cfg['in_dim']
        n_levels = cfg.get('n_levels', 2)

        net = []
        for i in range(n_levels):
            out_dim = in_dim // 2
            net.append(
                nn.Sequential(
                    UpBlock(
                        in_dim, out_dim, None,
                        up=cfg.get('up', 'conv'),
                        norm=cfg.get('norm'),
                        actv=cfg.get('up_actv', 'relu')
                    )
                )
            )
            in_dim = out_dim
        if cfg.get('pool'):
            self.up = 1
            self.pool = nn.AvgPool2d(2 ** n_levels)
        else:
            self.up = 2 ** n_levels
            self.pool = nn.Identity()
        self.net = nn.Sequential(*net)

        self.out_conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            get_norm(out_dim, None), get_actv(cfg.get('up_actv', 'relu')),
            nn.Conv2d(out_dim, 3, 1, 1),
        )

    def forward(self, feats):
        feats = self.pool(feats)
        feats = self.net(feats)
        rgb = self.out_conv(feats)
        return rgb
        

class UNetDecoder(nn.Module):

    def __init__(self, cfg):
        super(UNetDecoder, self).__init__()

        self.down_layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        in_dim = cfg['in_dim']
        self.n_levels = cfg.get('n_levels', 2)
        self.up = 1

        for i in range(self.n_levels):
            self.down_layers.append(
                DownBlock(
                    in_dim, in_dim, 
                    down=cfg.get('down', 'mean'),
                    norm=cfg.get('norm'),
                    actv=cfg.get('down_actv', 'leaky_relu')
                )
            )
            out_dim = in_dim // 2 ** (self.n_levels - i)
            self.skip_convs.append(nn.Conv2d(in_dim, out_dim, 1))
            self.up_layers.append(
                UpBlock(
                    out_dim * 2, out_dim, out_dim,
                    up=cfg.get('up', 'conv'),
                    norm=cfg.get('norm'),
                    actv=cfg.get('up_actv', 'relu')
                )
            )

        out_dim = in_dim // 2 ** self.n_levels
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            get_norm(out_dim, None), get_actv(cfg.get('up_actv', 'relu')),
            nn.Conv2d(out_dim, 3, 1, 1),
        )

    def forward(self, feats):
        skips = []
        for i in range(self.n_levels):
            skips.append(self.skip_convs[i](feats))
            feats = self.down_layers[i](feats)
        for i in range(self.n_levels - 1, -1, -1):
            feats = self.up_layers[i](feats, skips[i])
        rgb = self.out_conv(feats)
        return rgb


class PointCloudDecoder(nn.Module):

    def __init__(self, cfg):
        super(PointCloudDecoder, self).__init__()

        self.n_levels = cfg['n_levels']
        assert len(cfg['depth']) == self.n_levels
        assert len(cfg['dims']) == self.n_levels
        assert len(cfg['k']) == self.n_levels
        assert len(cfg['radius']) == self.n_levels
        self.up = 1

        self.blocks = nn.ModuleList()
        in_dim = cfg['in_dim']

        for i in range(self.n_levels):
            blocks = nn.ModuleList()
            for j in range(cfg['depth'][i]):
                if j < cfg['depth'][i] - 1:
                    blocks.append(
                        GCNBlock(
                            cfg['block'],
                            in_dim=in_dim, 
                            out_dim=in_dim,
                            scale_factor=1,
                            radius=cfg['radius'][i][j],
                            k=cfg['k'][i][j],
                            aggregate=cfg.get('aggregate', 'max'),
                            norm=cfg.get('norm'), 
                            actv=cfg.get('actv', 'relu'),
                            res=cfg.get('res')
                        )
                    )
                else:
                    blocks.append(
                        GCNUpBlock(
                            cfg['block'],
                            in_dim=in_dim, 
                            out_dim=cfg['dims'][i],
                            radius=cfg['radius'][i][j],
                            k=cfg['k'][i][j],
                            up=cfg.get('up', 'linear'),
                            aggregate=cfg.get('aggregate', 'max'),
                            norm=cfg.get('norm'), 
                            actv=cfg.get('actv', 'relu'),
                            res=cfg.get('res')
                        )
                    )
            in_dim = cfg['dims'][i]
            self.blocks.append(blocks)

        self.out_fc = nn.Conv1d(in_dim, 3, 1)

    def forward(self, xyz_list, feats):
        """
        Args:
            xyz_list (float tensor list): point coordinates at all levels.
            feats (float tensor, (bs, m, 3)): point features.
        
        Returns:
            rgb (float tensor, (bs, n, 3)): point RGB values.
        """
        assert len(xyz_list) == self.n_levels + 1, \
            ('[ERROR] number of points sets ({:d}) and number of levels ({:d}) '
             'must match'.format(len(xyz_list), self.n_levels + 1)
            )
        assert xyz_list[-1].size(1) == feats.size(2), \
            ('[ERROR] point cloud size ({:d}) and number of features ({:d}) '
             'must match'.format(xyz_list[-1].size(1), feats.size(2))
            )

        for i in range(self.n_levels):
            blocks = self.blocks[i]
            xyz, parent_xyz = xyz_list[-i - 1], xyz_list[-i - 2]
            for j in range(len(blocks)):
                _, feats = blocks[j](xyz, feats, parent_xyz)
                xyz = parent_xyz
        rgb = self.out_fc(feats)
        return rgb

###############################################################################
""" Stylizer """

class AdaIN(nn.Module):
    """ Adaptive instance normalization (Huang et al., ICCV 17) """

    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, c, s):
        mean = s.mean((-2, -1)).expand_as(c)
        std = s.std((-2, -1)).expand_as(c)
        cs = F.instance_norm(c) * std + mean
        return cs


class AdaAttN(nn.Module):
    """ Attention-weighted AdaIN (Liu et al., ICCV 21) """

    def __init__(self, qk_dim=None, v_dim=None, project=False):
        """
        Args:
            qk_dim (int): query and key size.
            v_dim (int): value size.
            project (int): apply projection to input features.
        """
        super(AdaAttN, self).__init__()

        if project:
            assert qk_dim is not None and v_dim is not None, \
                '[ERROR] qk_dim and v_dim must be given for feature projection'
            self.q_embed = nn.Conv1d(qk_dim, qk_dim, 1)
            self.k_embed = nn.Conv1d(qk_dim, qk_dim, 1)
            self.s_embed = nn.Conv1d(v_dim, v_dim, 1)
        else:
            self.q_embed = self.k_embed = self.s_embed = nn.Identity()

    def forward(self, q, k, c, s):
        """
        Args:
            q (float tensor, (bs, qk, *)): query (content) features.
            k (float tensor, (bs, qk, *)): key (style) features.
            c (float tensor, (bs, v, *)): content value features.
            s (float tensor, (bs, v, *)): style value features.

        Returns:
            cs (float tensor, (bs, v, *)): stylized content features.
        """
        shape = c.shape
        q, k = q.flatten(2), k.flatten(2)
        c, s = c.flatten(2), s.flatten(2)

        # QKV attention with projected content and style features
        q = self.q_embed(F.instance_norm(q)).transpose(2, 1)    # (bs, n, qk)
        k = self.k_embed(F.instance_norm(k))                    # (bs, qk, m)
        s = self.s_embed(s).transpose(2, 1)                     # (bs, m, v)
        attn = F.softmax(torch.bmm(q, k), -1)                   # (bs, n, m)
        
        # attention-weighted channel-wise statistics
        mean = torch.bmm(attn, s)                               # (bs, n, v)
        var = F.relu(torch.bmm(attn, s ** 2) - mean ** 2)       # (bs, n, v)
        mean = mean.transpose(2, 1)                             # (bs, v, n)
        std = torch.sqrt(var).transpose(2, 1)                   # (bs, v, n)
        
        cs = F.instance_norm(c) * std + mean                    # (bs, v, n)
        cs = cs.reshape(shape)
        return cs


class LST(nn.Module):

    def __init__(self, in_dim, embed_dim=32, n_layers=3):
        super(LST, self).__init__()

        self.embed_dim = embed_dim

        self.c_zipper = nn.Conv1d(in_dim, embed_dim, 1)
        self.c_unzipper = nn.Conv1d(embed_dim, in_dim, 1)

        c_net, s_net = [], []
        for i in range(n_layers - 1):
            out_dim = max(embed_dim, in_dim // 2)
            c_net.append(
                nn.Sequential(
                    nn.Conv1d(in_dim, out_dim, 1),
                    nn.ReLU(inplace=True),
                )
            )
            s_net.append(
                nn.Sequential(
                    nn.Conv1d(in_dim, out_dim, 1),
                    nn.ReLU(inplace=True),
                )
            )
            in_dim = out_dim
        c_net.append(nn.Conv1d(in_dim, embed_dim, 1))
        s_net.append(nn.Conv1d(in_dim, embed_dim, 1))
        self.c_net = nn.Sequential(*c_net)
        self.s_net = nn.Sequential(*s_net)

        self.c_fc = nn.Linear(embed_dim ** 2, embed_dim ** 2)
        self.s_fc = nn.Linear(embed_dim ** 2, embed_dim ** 2)

    def _vectorized_covariance(self, x):
        cov = torch.bmm(x, x.transpose(2, 1)) / x.size(-1)
        cov = cov.flatten(1)
        return cov

    def forward(self, c, s):
        c_shape = c.shape
        c, s = c.flatten(2), s.flatten(2)
        
        c_mean = c.mean(-1, keepdim=True)
        s_mean = s.mean(-1, keepdim=True)
        c = c - c_mean
        s = s - s_mean

        c_embed = self.c_net(c)
        c_cov = self._vectorized_covariance(c_embed)
        c_mat = self.c_fc(c_cov)
        c_mat = c_mat.reshape(-1, self.embed_dim, self.embed_dim)

        s_embed = self.s_net(s)
        s_cov = self._vectorized_covariance(s_embed)
        s_mat = self.s_fc(s_cov)
        s_mat = s_mat.reshape(-1, self.embed_dim, self.embed_dim)

        mat = torch.bmm(s_mat, c_mat)
        c = self.c_zipper(c)
        c = torch.bmm(mat, c)
        c = self.c_unzipper(c)
        cs = c + s_mean

        cs = cs.reshape(*c_shape)
        return cs


class AdaIN2DStylizer(nn.Module):
    """ Parameter-free AdaIN on image features (Huang et al., ICCV 17) """

    def __init__(self, cfg):
        super(AdaIN2DStylizer, self).__init__()

        vgg_dims = [64, 128, 256, 512, 512]
        vgg_layer = cfg['vgg_layer']
        self.vgg = NormalizedVGG(vgg_layer, pool=cfg.get('vgg_pool', 'max'))

        self.adain = AdaIN()

    def forward(self, style, feats):
        """
        Args:
            style (float tensor, (bs, 3, h, w)): style image.
            feats (float tensor, (bs, c, h, w)): content features.

        Returns:
            feats (float tensor, (bs, c, h, w)): transformed content features.
        """
        style = self.vgg(style)
        feats = self.adain(feats, style)
        return feats


class AdaAttN2DStylizer(nn.Module):
    """ AdaAttN stylizer (Liu et al., ICCV 21) """

    def __init__(self, cfg):
        super(AdaAttN2DStylizer, self).__init__()

        vgg_dims = [64, 128, 256, 512, 512]
        vgg_layers = sorted(cfg['vgg_layers'])
        assert vgg_layers[-1] <= 5, \
            ('[ERROR] VGG layers cannot exceed 5, '
             'got {:d}'.format(vgg_layers[-1])
            )
        self.vgg = NormalizedVGG(pool=cfg.get('vgg_pool', 'max'))

        for l in vgg_layers:
            qk_dim, v_dim = sum(vgg_dims[:l]), vgg_dims[l - 1]
            setattr(
                self, 'adaattn{:d}'.format(l), 
                AdaAttN(qk_dim, v_dim, project=True)
            )

    def forward(self, style, feats):
        """
        Args:
            style (float tensor, (bs, 3, h, w)): style image.
            feats (float tensor list): feature pyramid.

        Returns:
            feats (float tensor list): stylized feature pyramid.
        """
        style_feats = self.vgg(style)
        
        q, k = feats[0], style_feats[0]
        feats[0] = getattr(
            self, 'adaattn1', lambda q, k, c, s: c
        )(q, k, feats[0], style_feats[0])

        for i in range(1, len(feats)):
            q = F.interpolate(
                q, size=feats[i].shape[-2:], mode='bilinear',
                align_corners=False
            )
            k = F.interpolate(
                k, size=style_feats[i].shape[-2:], mode='bilinear',
                align_corners=False
            )
            q = torch.cat([q, feats[i]], 1)
            k = torch.cat([k, style_feats[i]], 1)
            feats[i] = getattr(
                self, 'adaattn{:d}'.format(i + 1), lambda q, k, c, s: c
            )(q, k, feats[i], style_feats[i])
        return feats


class Linear2DStylizer(nn.Module):
    """ Learned affine transform on image features (Li et al., CVPR 19) """

    def __init__(self, cfg):
        super(Linear2DStylizer, self).__init__()

        vgg_dims = [64, 128, 256, 512, 512]
        vgg_layer = cfg['vgg_layer']
        self.vgg = NormalizedVGG(vgg_layer, pool=cfg.get('vgg_pool', 'max'))

        self.lst = LST(
            in_dim=vgg_dims[vgg_layer - 1], 
            embed_dim=cfg.get('embed_dim', 32), 
            n_layers=cfg.get('n_embed_layers', 3)
        )

    def forward(self, style, feats):
        """
        Args:
            style (float tensor, (bs, 3, h, w)): style image.
            feats (float tensor, (bs, c, h, w)): content features.

        Returns:
            feats (float tensor, (bs, c, h, w)): transformed content features.
        """
        style = self.vgg(style)
        feats = self.lst(feats, style)
        return feats


class AdaIN3DStylizer(nn.Module):
    """ Learned AdaIN on point features (Huang et al., ICCV 17) """

    def __init__(self, cfg):
        super(AdaIN3DStylizer, self).__init__()

        vgg_dims = [64, 128, 256, 512, 512]
        vgg_layer = cfg['vgg_layer']
        self.vgg = NormalizedVGG(vgg_layer, pool=cfg.get('vgg_pool', 'max'))

        in_dim = vgg_dims[vgg_layer - 1]

        # content feature projection
        n_layers = cfg.get('n_zip_layers', 0)
        if n_layers > 0:
            zipper = [nn.Conv1d(in_dim, in_dim, 1)]
            unzipper = [nn.Conv1d(in_dim, in_dim, 1)]
            for i in range(n_layers - 1):
                zipper = zipper + [
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(in_dim, in_dim, 1),
                ]
                unzipper = [
                    nn.Conv1d(in_dim, in_dim, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                ] + unzipper
            self.zipper = nn.Sequential(*zipper)
            self.unzipper = nn.Sequential(*unzipper)
        else:
            self.zipper = self.unzipper = nn.Identity()

        self.adain = AdaIN()

        self.up = PointUpsample(mode=cfg.get('up', 'linear'))

    def forward(self, style, xyz_list, feats_list, up=True):
        """
        Args:
            style (float tensor, (bs, 3, h, w)): style image.
            xyz_list (float tensor list): point NDC coordinates at all levels.
            feats_list (float tensor list): point features at all levels.
            up (bool): if True, upsample output features to input resolution.

        Returns:
            feats (float tensor, (bs, c, n)): transformed content features.
        """
        style = self.vgg(style)
        feats = self.zipper(feats_list[-1])
        feats = self.adain(feats, style)
        feats = self.unzipper(feats)
        output_dict = {'feats': feats}
        
        # upsample features to match the resolution of input point cloud
        if up:
            for i in range(len(xyz_list) - 2, -1, -1):
                feats = self.up(xyz_list[i + 1], xyz_list[i], feats)
            output_dict['up_feats'] = feats
        return output_dict


class AdaAttN3DStylizer(nn.Module):
    """ Attention-weighted AdaIN stylizer (Liu et al., ICCV 21) """

    def __init__(self, cfg):
        super(AdaAttN3DStylizer, self).__init__()

        vgg_dims = [64, 128, 256, 512, 512]
        vgg_layer = cfg['vgg_layer']
        self.pyramid = cfg.get('pyramid')

        v_dim = vgg_dims[vgg_layer - 1]
        if self.pyramid:
            qk_dim = sum(vgg_dims[:vgg_layer])
            self.vgg = NormalizedVGG(pool=cfg.get('vgg_pool', 'max'))
        else:
            qk_dim = v_dim
            self.vgg = NormalizedVGG(vgg_layer, pool=cfg.get('vgg_pool', 'max'))
        n_layers = cfg.get('n_zip_layers', 0)
        
        self.adaattn = AdaAttN(qk_dim, v_dim, project=True)

        # transform content features to match VGG feature space
        if n_layers > 0:
            q_zipper = [nn.Conv1d(qk_dim, qk_dim, 1)]
            v_zipper = [nn.Conv1d(v_dim, v_dim, 1)]
            v_unzipper = [nn.Conv1d(v_dim, v_dim, 1)]
            for i in range(n_layers - 1):
                q_zipper = q_zipper + [
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(qk_dim, qk_dim, 1),
                ]
                v_zipper = v_zipper + [
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(v_dim, v_dim, 1),
                ]
                v_unzipper = [
                    nn.Conv1d(v_dim, v_dim, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                ] + v_unzipper
            self.q_zipper = nn.Sequential(*q_zipper)
            self.v_zipper = nn.Sequential(*v_zipper)
            self.v_unzipper = nn.Sequential(*v_unzipper)
        else:
            self.q_zipper = self.v_zipper = self.v_unzipper = nn.Identity()

        self.down = PointDownsample(mode=cfg.get('down', 'linear'))
        self.up = PointUpsample(mode=cfg.get('up', 'linear'))

    def forward(self, style, xyz_list, feats_list, up=True):
        """
        Args:
            style (float tensor, (bs, 3, h, w)): style image.
            xyz_list (float tensor list): point NDC coordinates at all levels.
            feats_list (float tensor list): point features at all levels.
            up (bool): if True, upsample output features to input resolution.

        Returns:
            feats (float tensor, (bs, c, n)): transformed content features.
        """
        # pyramidal query features
        if self.pyramid:
            style = self.vgg(style)
            q, k = feats_list[0], style[0]
            for i in range(len(feats_list) - 1):
                _, q = self.down(xyz_list[i], q, xyz_list[i + 1])
                k = F.interpolate(
                    k, size=style[i + 1].shape[-2:], mode='bilinear',
                    align_corners=False
                )
                q = torch.cat([q, feats_list[i + 1]], 1)
                k = torch.cat([k, style[i + 1]], 1)
            c = feats_list[-1]
            s = style[len(feats_list) - 1]
        else:
            k = s = self.vgg(style)
            q = c = feats_list[-1]

        q, c = self.q_zipper(q), self.v_zipper(c)
        cs = self.adaattn(q, k, c, s)
        feats = self.v_unzipper(cs)
        output_dict = {'feats': feats}

        # upsample features to match the resolution of input point cloud
        if up:
            for i in range(len(xyz_list) - 2, -1, -1):
                feats = self.up(xyz_list[i + 1], xyz_list[i], feats)
            output_dict['up_feats'] = feats
        return output_dict


class Linear3DStylizer(nn.Module):
    """ Learned affine transform on point features (Li et al., CVPR 19) """

    def __init__(self, cfg):
        super(Linear3DStylizer, self).__init__()

        vgg_dims = [64, 128, 256, 512, 512]
        vgg_layer = cfg['vgg_layer']
        self.vgg = NormalizedVGG(vgg_layer, pool=cfg.get('vgg_pool', 'max'))

        self.lst = LST(
            in_dim=vgg_dims[vgg_layer - 1], 
            embed_dim=cfg.get('embed_dim', 32), 
            n_layers=cfg.get('n_embed_layers', 3)
        )

        self.up = PointUpsample(mode=cfg.get('up', 'linear'))

    def forward(self, style, xyz_list, feats_list, up=True):
        """
        Args:
            style (float tensor, (bs, 3, h, w)): style image.
            xyz_list (float tensor list): point NDC coordinates at all levels.
            feats_list (float tensor list): point features at all levels.
            up (bool): if True, upsample output features to input resolution.

        Returns:
            feats (float tensor, (bs, c, n)): transformed content features.
        """
        style = self.vgg(style)
        feats = self.lst(feats_list[-1], style)
        output_dict = {'feats': feats}

        # upsample features to match the resolution of input point cloud
        if up:
            for i in range(len(xyz_list) - 2, -1, -1):
                feats = self.up(xyz_list[i + 1], xyz_list[i], feats)
            output_dict['up_feats'] = feats
        return output_dict

###############################################################################
""" Discriminator """

class PatchDiscriminator(nn.Module):
    """ PatchGAN discriminator (Wiles et al., CVPR 20) """

    def __init__(self, cfg):
        super(PatchDiscriminator, self).__init__()

        self.conv1 = nn.Sequential(
            get_conv2d(
                3, 64, 4, 2, 1, spectral_norm=cfg.get('spectral_norm')
            ),
            get_actv(cfg.get('actv', 'leaky_relu')),
        )
        self.conv2 = nn.Sequential(
            get_conv2d(
                64, 128, 4, 2, 1, spectral_norm=cfg.get('spectral_norm')
            ),
            get_actv(cfg.get('actv', 'leaky_relu')),
        )
        self.conv3 = nn.Sequential(
            get_conv2d(
                128, 256, 4, 2, 1, spectral_norm=cfg.get('spectral_norm')
            ),
            get_actv(cfg.get('actv', 'leaky_relu')),
        )
        self.conv4 = nn.Sequential(
            get_conv2d(
                256, 512, 4, 1, 1, spectral_norm=cfg.get('spectral_norm')
            ),
            get_actv(cfg.get('actv', 'leaky_relu')),
        )
        self.conv5 = get_conv2d(
            512, 1, 4, 1, 1, spectral_norm=cfg.get('spectral_norm')
        )

    def forward(self, x):
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)
        out = self.conv5(h4)
        return [h1, h2, h3, h4, out]