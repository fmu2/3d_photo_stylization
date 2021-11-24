import torch
import torch.nn as nn

from .module import *
from .pointnet2 import PointUpsample, PointDownsample
from .render import view2ndc, ndc2view

import matplotlib.pyplot as plt


class AblationModel(nn.Module):

    def __init__(self, enc_cfg, dec_cfg):
        super(AblationModel, self).__init__()

        self.stylization = False            # whether in stylization mode

        # point cloud utilities
        ## NOTE: all geometry-based, no learnable parameters
        self.unprojector = Unprojector()
        self.view_transformer = ViewTransformer()
        self.renderer = Renderer()
        
        # encoder
        self.encoder = VGGEncoder(enc_cfg['vgg'])
        self.pt_down = PointDownsample(scale_factor=8, mode='linear')
        self.pt_up = PointUpsample()

        # decoder
        self.decoder = UNetDecoder(dec_cfg['unet'])

    def convert_for_stylization(self, cfg, freeze_enc=True, freeze_dec=True):
        if freeze_enc:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if freeze_dec:
            for p in self.decoder.parameters():
                p.requires_grad = False

        # stylizer
        self.stylizer = AdaAttN3DStylizer(cfg['adaattn'])

        self.stylization = True
        params = [p for p in self.parameters() if p.requires_grad]
        return params

    def forward(self, input_dict, h=224, w=None, src_h=448, src_w=448,
                ndc=True, pcd_size=None, pcd_scale=1, anti_aliasing=True, rgb_only=False):
        """
        Args:
            input_dict (dict):
                K (float tensor, (bs, 5)): camera intrinsics (fov, fx, fy, cx, cy).
                Ms (float tensor, (bs, v, 3, 4)): camera extrinsics (R | t).
                src_rgb (float tensor, (bs, 3, p)): point RGB.
                src_z (float tensor, (bs, p, 1)): point depth.
                src_uv (float tensor, (bs, p, 3)): point uv-coordinates.
                n_pts ((optional) int tensor, (bs,)): number of points.
                tgt_fovs ((optional) float tensor, (bs, v)): target-view FOVs.
                style ((optional) float tensor, (bs, 3, h, w)): style images.
            h (int): height of rendered images.
            w (int): width of rendered images.
            ndc (bool): if True, construct point cloud in NDC space.
            pcd_size (int): number of points to draw for point cloud processing.
            pcd_scale (float): point cloud scale.
            anti_aliasing (bool): if True, apply anti-aliasing.

        Returns:
            output_dict (dict):
                enc_feats (float tensor, (bs, v, c, hc, wc)): encoder features.
                pred_rgb (float tensor, (bs, v, 3, h, w)): predicted images.
                tgt_rgb (float tensor, (bs, v, 3, h, w)): target images.
                uv ((optional) float tensor, (bs, v, p, 2)): points uv-coordinates.
                viz ((optional) bool tensor, (bs, v, p)): point visibility.
        """
        # parse input
        K = input_dict['K']
        bs, fov = K.size(0), K[:, 0]
        extreme_Ms, Ms = input_dict['extreme_Ms'], input_dict['Ms']
        n_extreme_views, n_views = extreme_Ms.size(1), Ms.size(1)
        tgt_fovs = input_dict.get(
            'tgt_fovs', fov.unsqueeze(-1).repeat(1, n_views)
        )
        if w is None:
            w = h
        style = input_dict.get('style')
        
        # prepare point cloud
        rgb = input_dict['src_rgb']
        uv, z = input_dict['src_uv'], input_dict['src_z']
        xyz = self.unprojector(uv, z, K)

        # rasterize to extreme views
        proj_rgb_list, proj_z_list, mask_list = [], [], []
        for k in range(n_extreme_views):
            new_xyz = self.view_transformer(xyz, extreme_Ms[:, k])
            rgbz = torch.cat([rgb, new_xyz[..., -1].unsqueeze(1)], 1)
            tgt_dict = self.renderer(
                new_xyz, rgbz, fov, h=src_h, w=src_w,
                anti_aliasing=False, denoise=True
            )
            proj_rgb_list.append(tgt_dict['data'][:, :3])
            proj_z_list.append(tgt_dict['data'][:, -1:])
            mask_list.append(tgt_dict['mask'][:, -1:])
        proj_rgb = torch.stack(proj_rgb_list, 1).flatten(0, 1)      # (bs * v, 3, h, w)
        proj_z = torch.stack(proj_z_list, 1).flatten(0, 1)          # (bs * v, 1, h, w)
        mask = torch.stack(mask_list, 1).flatten(0, 1)              # (bs * v, 1, h, w)

        # feature extraction
        feats = self.encoder(proj_rgb)                              # (bs * v, c, h / 4, w / 4)
        proj_h, proj_w = feats.shape[-2:]
        n_pts = proj_h * proj_w
        feats = feats.flatten(-2).transpose(-1, -2)                 # (bs * v, p, c)
        feats = feats.reshape(bs, n_extreme_views, n_pts, -1)       # (bs, v, p, c)
        feats = feats.flatten(1, 2)                                 # (bs, v * p, c)

        # back-project extreme-view pixels to 3D
        v_grid, u_grid = torch.meshgrid(
            torch.arange(proj_h, device=rgb.device) + 0.5, 
            torch.arange(proj_w, device=rgb.device) + 0.5
        )
        proj_uv = torch.stack([u_grid, v_grid], -1)                 # (h, w, 2)
        proj_uv = proj_uv.repeat(bs * n_extreme_views, 1, 1, 1)     # (bs * v, h, w, 2)
        
        proj_uv[..., 0] = proj_uv[..., 0] / proj_w * 2 - 1
        proj_uv[..., 1] = proj_uv[..., 1] / proj_h * 2 - 1
        proj_z = F.grid_sample(proj_z, proj_uv, mode='nearest')
        mask = F.grid_sample(mask.float(), proj_uv, mode='nearest') > 0

        proj_uv = proj_uv.reshape(bs * n_extreme_views, n_pts, 2)   # (bs * v, p, 2)
        proj_uv = (proj_uv + 1) / 2
        proj_uv[..., 0] *= src_w
        proj_uv[..., 1] *= src_h
        proj_z = proj_z.flatten(-2).transpose(-1, -2)               # (bs * v, p, 1)
        bp_K = K.repeat_interleave(n_extreme_views, dim=0)
        bp_xyz = self.unprojector(proj_uv, proj_z, bp_K)            # (bs * v, p, 3)
        bp_xyz = bp_xyz.reshape(bs, n_extreme_views, n_pts, 3)      # (bs, v, p, 3)

        mask = mask.flatten(-2).transpose(-1, -2)                   # (bs * v, p, 1)
        mask = mask.reshape(bs, n_extreme_views, n_pts, 1)          # (bs, v, p, 1)
        mask = mask.flatten(1, 2)                                   # (bs, v * p, 1)
        n_valid_pts = mask.squeeze(-1).sum(-1).tolist()
        n_pts = min(n_valid_pts)
        
        # back-project to point cloud
        bp_xyz_list = []
        for k in range(n_extreme_views):
            new_xyz = self.view_transformer(
                bp_xyz[:, k], extreme_Ms[:, k], inverse=True
            )
            bp_xyz_list.append(new_xyz)
        bp_xyz = torch.cat(bp_xyz_list, 1)                          # (bs, v * p, 3)
        bp_xyz = bp_xyz.masked_select(mask).reshape(-1, 3).split(n_valid_pts)
        feats = feats.masked_select(mask).reshape(-1, feats.size(-1)).split(n_valid_pts)

        bp_xyz_list, feats_list = [], []
        for i in range(bs):
            idx = torch.randperm(n_valid_pts[i], device=xyz.device)
            bp_xyz_list.append(bp_xyz[i][idx][:n_pts])
            feats_list.append(feats[i][idx][:n_pts])
        bp_xyz = torch.stack(bp_xyz_list)                           # (bs, p, 3)
        feats = torch.stack(feats_list).transpose(-1, -2)           # (bs, c, p)

        n_pts = input_dict.get(
            'n_pts', xyz.new_ones(xyz.size(0), dtype=torch.int) * xyz.size(1)
        )
        if pcd_size is None:
            pcd_size = n_pts.min()
        assert pcd_size <= n_pts.min(), \
            ('[ERROR] point sample size ({:d}) cannot exceed smallest '
             'point cloud size ({:d})'.format(pcd_size, n_pts.min())
            )
        
        raw_xyz, raw_rgb = xyz, rgb
        ## NOTE: assume that the ordering of points has been randomized
        ## (e.g., by random shuffling).
        xyz, rgb = xyz[:, :pcd_size], rgb[..., :pcd_size]

        # view space -> NDC space
        ## NOTE: assume that valid points have depth < 1e5.
        if ndc:
            z = xyz[..., 2]
            near = 0.99 * z.amin(1)
            far = (z * (z < 1e5)).quantile(0.95, 1)
            far = torch.maximum(far, near * 2)
            ## NOTE: preserve aspect ratio by setting ar=1
            xyz_ndc = view2ndc(xyz, near, far, fov, ar=1) * pcd_scale
            bp_xyz_ndc = view2ndc(bp_xyz, near, far, fov, ar=1) * pcd_scale
        else:
            xyz_ndc, bp_xyz_ndc = xyz, bp_xyz
        
        # point down-/up-sampling
        bp_xyz_ndc, feats = self.pt_down(bp_xyz_ndc, feats)
        xyz_ndc_list = [xyz_ndc, bp_xyz_ndc]
        feats_list = [feats]
        up_feats = None
        if not self.stylization:
            up_feats = self.pt_up(bp_xyz_ndc, xyz_ndc, feats)

        # stylization
        if self.stylization:
            assert style is not None, '[ERROR] style image is missing'
            style_dict = self.stylizer(style, xyz_ndc_list, feats_list)
            up_feats = style_dict['up_feats']

        pred_feats_list = []
        tgt_rgb_list, pred_rgb_list = [], []
        uv_list, viz_list = [], []
        output_dict = dict()
        
        # rasterization / reconstruction
        for k in range(n_views):
            tgt_fov = tgt_fovs[:, k]
            new_xyz = self.view_transformer(xyz, Ms[:, k])
            new_raw_xyz = self.view_transformer(raw_xyz, Ms[:, k])

            # 1) rasterize featurized point cloud to novel view
            # 2) decode 2D feature maps into RGB image
            pred_dict = self.renderer(
                new_xyz, up_feats, 
                tgt_fov, h, w, anti_aliasing=anti_aliasing, return_uv=True
            )
            pred_feats = pred_dict['data']
            pred_rgb = self.decoder(pred_feats)
            pred_rgb_list.append(pred_rgb)
            if not rgb_only:
                pred_feats_list.append(pred_feats)
                uv_list.append(pred_dict['uv'])
                viz_list.append(pred_dict['viz'])
            
            # rasterize RGB point cloud
            tgt_dict = self.renderer(
                new_raw_xyz, raw_rgb, tgt_fov, h, w, 
                anti_aliasing=anti_aliasing, denoise=True
            )
            tgt_rgb_list.append(tgt_dict['data'])

        pred_rgb = torch.stack(pred_rgb_list, 1)            # (bs, v, 3, h, w)
        tgt_rgb = torch.stack(tgt_rgb_list, 1)              # (bs, v, 3, h, w)
        output_dict['pred_rgb'] = pred_rgb
        output_dict['tgt_rgb'] = tgt_rgb

        if not rgb_only:
            pred_feats = torch.stack(pred_feats_list, 1)
            output_dict['pred_feats'] = pred_feats
            if n_views > 1:
                uv = torch.stack(uv_list, 1)                # (bs, v, p, 2)
                viz = torch.stack(viz_list, 1)              # (bs, v, p)
                output_dict['uv'] = uv
                output_dict['viz'] = viz

        return output_dict