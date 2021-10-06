import torch
import torch.nn as nn

from .module import *
from .render import view2ndc


class Model3D(nn.Module):

    def __init__(self, enc_cfg, dec_cfg):
        super(Model3D, self).__init__()

        self.stylization = False           # whether in stylization mode
        self.render_then_decode = True     # whether to render 2D feature maps before decoding

        # point cloud utilities
        ## NOTE: all geometry-based, no learnable parameters
        self.unprojector = Unprojector()
        self.view_transformer = ViewTransformer()
        self.renderer = Renderer()
        
        # encoder
        self.encoder = PointCloudEncoder(enc_cfg['pcd'])

        # decoder
        if dec_cfg['arch'] == 'unet':
            self.decoder = UNetDecoder(dec_cfg['unet'])
        elif dec_cfg['arch'] == 'pcd':
            self.decoder = PointCloudDecoder(dec_cfg['pcd'])
            self.render_then_decode = False
        else:
            raise NotImplementedError(
                '[ERROR] invalid decoder: {:s}'.format(dec_cfg['arch'])
            )

    def convert_for_stylization(self, cfg, freeze_enc=True, freeze_dec=True):
        if freeze_enc:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if freeze_dec:
            for p in self.decoder.parameters():
                p.requires_grad = False

        # stylizer
        if cfg['arch'] == 'adain':
            self.stylizer = AdaIN3DStylizer(cfg['adain'])
        elif cfg['arch'] == 'adaattn':
            self.stylizer = AdaAttN3DStylizer(cfg['adaattn'])
        elif cfg['arch'] == 'linear':
            self.stylizer = Linear3DStylizer(cfg['linear'])
        else:
            raise NotImplementedError(
                '[ERROR] invalid stylizer: {:s}'.format(cfg['arch'])
            )
        
        self.stylization = True
        params = [p for p in self.parameters() if p.requires_grad]
        return params

    def _view2ndc(self, xyz, fov, ar=1, scale=1):
        # view space -> NDC space
        ## NOTE: assume that valid points have depth < 1e5.
        z = xyz[..., 2]
        near = 0.99 * z.amin(1)
        far = (z * (z < 1e5)).quantile(0.95, 1)
        far = torch.maximum(far, near * 2)
        xyz_ndc = view2ndc(xyz, near, far, fov, ar) * scale
        return xyz_ndc


    def forward(self, input_dict, h=224, w=None, pcd_size=None, pcd_scale=None):
        """
        Args:
            input_dict (dict):
                K (float tensor, (bs, 5)): camera intrinsics (fov, fx, fy, cx, cy).
                Ms (float tensor, (bs, v, 3, 4)): camera extrinsics (R | t).
                src_rgb (float tensor, (bs, 3, p)): point RGB.
                src_z (float tensor, (bs, p, 1)): point depth.
                src_uv (float tensor, (bs, p, 3)): point uv-coordinates.
                n_pts ((optional) int tensor, (bs,)): number of points.
                tgt_fovs ((optional) float tensor, (bs,)): target-view FOVs.
                style ((optional) float tensor, (bs, 3, h, w)): style images.
            h (int): height of rendered images.
            w (int): width of rendered images.
            pcd_size (int): number of points to draw for point cloud processing.
            pcd_scale (float): point cloud scale.

        Returns:
            output_dict (dict):
                enc_feats (float tensor, (bs, v, c, hc, wc)): encoder features.
                pred_rgb (float tensor, (bs, v, 3, h, w)): predicted images.
                tgt_rgb (float tensor, (bs, v, 3, h, w)): target images.
                uv ((optional) float tensor, (bs, v, p, 2)): points uv-coordinates.
                viz ((optional) bool tensor, (bs, v, p)): point visibility.
        """
        # parse input
        K, Ms = input_dict['K'], input_dict['Ms']
        fov = K[:, 0]
        n_views = Ms.size(1)
        rgb = input_dict['src_rgb']
        uv, z = input_dict['src_uv'], input_dict['src_z']
        n_pts = input_dict.get(
            'n_pts', uv.new_ones(uv.size(0), dtype=torch.int) * uv.size(1)
        )
        if pcd_size is None:
            pcd_size = n_pts.min()
        assert pcd_size <= n_pts.min(), \
            ('[ERROR] point sample size ({:d}) cannot exceed smallest '
             'point cloud size ({:d})'.format(pcd_size, n_pts.min())
            )
        if pcd_scale is None:
            pcd_scale = 1
        style = input_dict.get('style')
        tgt_fovs = input_dict.get(
            'tgt_fovs', fov.unsqueeze(-1).repeat(1, n_views)
        )
        if w is None:
            w = h
        output_dict = dict()
        
        # prepare point cloud
        xyz = self.unprojector(uv, z, K)
        raw_xyz, raw_rgb = xyz, rgb
        ## NOTE: assume that the ordering of points has been randomized
        ## (e.g., by random shuffling).
        xyz, rgb = xyz[:, :pcd_size], rgb[..., :pcd_size]
        ## NOTE: preserve aspect ratio by setting ar=1
        xyz_ndc = self._view2ndc(xyz, fov, ar=1, scale=pcd_scale)

        # feature extraction
        enc_dict = self.encoder(xyz_ndc, rgb, up=(not self.stylization))
        xyz_ndc_list = enc_dict['xyz_list']
        feats, up_feats = enc_dict['feats'], enc_dict.get('up_feats')

        # stylization
        if self.stylization:
            assert style is not None, '[ERROR] style image is missing'
            up_feats = self.stylizer(style, xyz_ndc_list, feats)

        pred_feats_list = [] 
        tgt_rgb_list, pred_rgb_list = [], []
        uv_list, viz_list = [], []
        
        # rasterization / reconstruction
        for k in range(n_views):
            fov = tgt_fovs[:, k]
            new_xyz = self.view_transformer(xyz, Ms[:, k])
            new_raw_xyz = self.view_transformer(raw_xyz, Ms[:, k])

            if self.render_then_decode:
                # 1) rasterize featurized point cloud to novel view
                # 2) decode 2D feature maps into RGB image
                pred_dict = self.renderer(
                    new_xyz, up_feats, fov, h, w, return_uv=True
                )
                pred_feats = pred_dict['data']
                pred_rgb = self.decoder(pred_feats)
            else:
                # 1) decode featurized point cloud into RGB point cloud
                # 2) rasterize RGB point cloud to novel view
                rgb = self.decoder(xyz_ndc_list, feats)
                pred_dict = self.renderer(
                    new_xyz, torch.cat([rgb, up_feats], 1), fov, h, w, 
                    return_uv=True
                )
                pred_feats = pred_dict['data'][:, 3:]
                pred_rgb = pred_dict['data'][:, :3]

            pred_feats_list.append(pred_feats)
            pred_rgb_list.append(pred_rgb)
            uv_list.append(pred_dict['uv'])
            viz_list.append(pred_dict['viz'])
            
            # rasterize RGB point cloud
            tgt_dict = self.renderer(
                new_raw_xyz, raw_rgb, fov, h, w, denoise=True
            ) 
            tgt_rgb_list.append(tgt_dict['data'])

        pred_feats = torch.stack(pred_feats_list, 1)    # (bs, v, c, hc, wc)
        pred_rgb = torch.stack(pred_rgb_list, 1)        # (bs, v, 3, h, w)
        tgt_rgb = torch.stack(tgt_rgb_list, 1)          # (bs, v, 3, h, w)
        output_dict['pred_feats'] = pred_feats
        output_dict['pred_rgb'] = pred_rgb
        output_dict['tgt_rgb'] = tgt_rgb
        if n_views > 1:
            uv = torch.stack(uv_list, 1)                # (bs, v, p, 2)
            viz = torch.stack(viz_list, 1)              # (bs, v, p)
            output_dict['uv'] = uv
            output_dict['viz'] = viz

        return output_dict


class Model2D(nn.Module):

    def __init__(self, enc_cfg, dec_cfg):
        super(Model2D, self).__init__()

        self.stylization = False
        self.return_pyramid = False

        # encoder
        self.encoder = VGGEncoder(enc_cfg['vgg'])

        # decoder
        if dec_cfg['arch'] == 'vgg':
            self.decoder = VGGDecoder(dec_cfg['vgg'])
        elif dec_cfg['arch'] == 'vgg_attn':
            self.decoder = VGGAttNDecoder(dec_cfg['vgg_attn'])
            self.return_pyramid = True
        else:
            raise NotImplementedError(
                '[ERROR] invalid decoder: {:s}'.format(dec_cfg['arch'])
            )

    def convert_for_stylization(self, cfg, freeze_enc=True, freeze_dec=True):
        if freeze_enc:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if freeze_dec:
            for param in self.decoder.parameters():
                param.requires_grad = False

        # stylizer
        if cfg['arch'] == 'adain':
            self.stylizer = AdaIN2DStylizer(cfg['adain'])
        elif cfg['arch'] == 'adaattn':
            self.stylizer = AdaAttN2DStylizer(cfg['adaattn'])
            self.return_pyramid = True
        elif cfg['arch'] == 'linear':
            self.stylizer = Linear2DStylizer(cfg['linear'])
        else:
            raise NotImplementedError(
                '[ERROR] invalid stylizer: {:s}'.format(cfg['arch'])
            )
        
        self.stylization = True
        params = [p for p in self.parameters() if p.requires_grad]
        return params

    def forward(self, input_dict):
        """
        Args:
            input_dict (dict):
                rgb (float tensor, (bs, 3, h, w)): input images.
                style ((optional) float tensor, (bs, 3, h, w)): style images.

        Returns:
            output_dict (dict):
                pred_rgb (float tensor, (bs, 3, h, w)): predicted images.
                tgt_rgb (float tensor, (bs, 3, h, w)): target images.
        """
        output_dict = dict()
        rgb, style = input_dict['src_rgb'], input_dict.get('style')

        # feature extraction
        feats = self.encoder(rgb, return_pyramid=self.return_pyramid)

        # stylization
        if self.stylization:
            assert style is not None, '[ERROR] style image is missing'
            feats = self.stylizer(style, feats)

        # reconstruction
        pred_rgb = self.decoder(feats)

        output_dict['pred_feats'] = feats
        output_dict['pred_rgb'] = pred_rgb

        return output_dict