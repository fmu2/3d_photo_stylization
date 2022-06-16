import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import PatchDiscriminator
from .model import Model3D, Model2D
from .loss import InpaintingLoss, StylizationLoss, GANLoss
from .metric import RMSE, PSNR, SSIM


class InpaintingTrainer():

    def __init__(self, netG, netD, 
                 optimG, optimD, schedG, schedD,
                 render_loss, G_GAN_loss, D_GAN_loss, weights):

        self.parallel = False

        self.netG = netG  # generator (inpainting model)
        self.netD = netD  # discriminator
        
        self.optimG = optimG
        self.optimD = optimD
        self.schedG = schedG
        self.schedD = schedD

        self.render_loss = render_loss
        self.G_GAN_loss = G_GAN_loss
        self.D_GAN_loss = D_GAN_loss
        assert len(weights) == 3, \
            '[ERROR] expect three weights, got {:d}'.format(len(weights))
        self.weights = weights

        self.rmse = RMSE()
        self.psnr = PSNR()
        self.ssim = SSIM()

    def load(self, ckpt):
        if self.parallel:
            self.netG.module.load_state_dict(ckpt['netG'], strict=False)
            self.netD.module.load_state_dict(ckpt['netD'])
        else:
            self.netG.load_state_dict(ckpt['netG'], strict=False)
            self.netD.load_state_dict(ckpt['netD'])

        self.optimG.load_state_dict(ckpt['optimG'])
        self.optimD.load_state_dict(ckpt['optimD'])
        self.schedG.load_state_dict(ckpt['schedG'])
        self.schedD.load_state_dict(ckpt['schedD'])

    def save(self, config, itr):
        if self.parallel:
            netG_state = self.netG.module.state_dict()
            netD_state = self.netD.module.state_dict()
        else:
            netG_state = self.netG.state_dict()
            netD_state = self.netD.state_dict()

        ckpt = {
            'config': config,
            'itr': itr,
            'netG': netG_state,
            'netD': netD_state,
            'optimG': self.optimG.state_dict(),
            'optimD': self.optimD.state_dict(),
            'schedG': self.schedG.state_dict(),
            'schedD': self.schedD.state_dict(),
        }
        return ckpt

    def cuda(self, parallel=False):
        self.netG.cuda()
        self.netD.cuda()
        self.render_loss.cuda()
        self.G_GAN_loss.cuda()
        self.D_GAN_loss.cuda()
        self.rmse.cuda()
        self.psnr.cuda()
        self.ssim.cuda()
        if parallel:
            self.netG = nn.DataParallel(self.netG)
            self.netD = nn.DataParallel(self.netD)
            self.render_loss.data_parallel()
            self.parallel = True

    def train(self):
        self.netG.train()
        self.netD.train()

    def eval(self):
        self.netG.eval()
        self.netD.eval()

    def run(self, input_dict, h=None, w=None, mode='train', 
                nvs=True, ndc=True, pcd_size=None):
        for k in input_dict.keys():
            input_dict[k] = input_dict[k].cuda(non_blocking=True)

        if nvs:
            output_dict = self.netG(input_dict, h, w, ndc, pcd_size)
            pred_rgb = output_dict['pred_rgb']                          # (bs, v, 3, h, w)
            bs, n_views = pred_rgb.shape[:2]
            pred_rgb = pred_rgb.flatten(0, 1)                           # (bs * v, 3, h, w)
            pred_feats = output_dict['pred_feats'].flatten(0, 1)        # (bs * v, c, hc, wc)

            # find correspondences across views
            match = viz_mtx = None
            if 'uv' in output_dict:
                uv, viz = output_dict['uv'], output_dict['viz']         # (bs, v, p)
                # multi-view correspondences
                match = F.grid_sample(
                    pred_rgb, uv.flatten(0, 1).unsqueeze(-2), 
                    align_corners=False
                ).squeeze(-1).reshape(bs, n_views, 3, -1)               # (bs, v, 3, p)
                match = match.permute(0, 3, 1, 2).flatten(0, 1)         # (bs * p, v, 3)
                # pairwise visibility matrix
                viz_mtx = viz.unsqueeze(1) * viz.unsqueeze(2)           # (bs, v, v, p)
                viz_mtx = viz_mtx.permute(0, 3, 1, 2).flatten(0, 1)     # (bs * p, v, v)
                viz_mtx = viz_mtx * ~torch.eye( # ignore self-pairings
                    viz_mtx.size(-1), device=viz_mtx.device, 
                    dtype=torch.bool
                )

            if 'tgt_rgb' in input_dict:
                tgt_rgb = input_dict['tgt_rgb'].flatten(0, 1)
            else:
                tgt_rgb = output_dict['tgt_rgb'].flatten(0, 1)

            if tgt_rgb.shape[-2:] != pred_rgb.shape[-2:]:
                tgt_rgb = F.interpolate(
                    tgt_rgb, pred_rgb.shape[-2:], mode='bilinear', 
                    align_corners=False
                )
        else:
            output_dict = self.netG(input_dict)
            pred_feats = output_dict['pred_feats']
            pred_rgb = output_dict['pred_rgb']
            tgt_rgb = input_dict['src_rgb']
            match = viz_mtx = None

        # generator loss
        loss_dict = self.render_loss(
            pred_rgb, tgt_rgb, pred_feats, match, viz_mtx
        )
        render_loss = loss_dict['total']
        G_loss = self.weights[0] * render_loss
        
        if self.weights[1] > 0:
            fake = self.netD(pred_rgb)
            G_GAN_loss = self.G_GAN_loss(fake, is_real=False, is_critic=False)
            loss_dict['ganG'] = G_GAN_loss
            G_loss = G_loss + self.weights[1] * G_GAN_loss 

        if mode == 'train':
            self.optimG.zero_grad()
            G_loss.backward()
            self.optimG.step()
            self.schedG.step()

        # discriminator loss
        if self.weights[2] > 0:
            real = self.netD(tgt_rgb)
            fake = self.netD(pred_rgb.detach())
            D_real_loss = self.D_GAN_loss(real, is_real=True, is_critic=True)
            D_fake_loss = self.D_GAN_loss(fake, is_real=False, is_critic=True)
            D_GAN_loss = (D_real_loss + D_fake_loss) / 2
            loss_dict['ganD'] = D_GAN_loss
            D_loss = self.weights[2] * D_GAN_loss

            if mode == 'train':
                self.optimD.zero_grad()
                D_loss.backward()
                self.optimD.step()
                self.schedD.step()     

        loss_dict.pop('total')
        for k in loss_dict.keys():
            loss_dict[k] = loss_dict[k].cpu()

        pred_rgb = torch.clamp(pred_rgb.detach(), 0, 1)
        output_dict = {
            'pred': pred_rgb.cpu(),
            'target': tgt_rgb.cpu(),
        }
        
        metric_dict = {
            'rmse': self.rmse(pred_rgb, tgt_rgb).cpu(),
            'psnr': self.psnr(pred_rgb, tgt_rgb).cpu(),
            'ssim': self.ssim(pred_rgb, tgt_rgb).cpu(),
        }

        return output_dict, loss_dict, metric_dict


class StylizationTrainer():
    
    def __init__(self, net, optim, sched, loss):

        self.parallel = False

        self.net = net
        self.optim = optim
        self.sched = sched
        self.loss = loss

    def load(self, ckpt):
        net_state = ckpt['net'] if 'net' in ckpt.keys() else ckpt['netG']
        if self.parallel:
            self.net.module.load_state_dict(net_state, strict=False)
        else:
            self.net.load_state_dict(net_state, strict=False)

        if 'optim' in ckpt.keys():
            self.optim.load_state_dict(ckpt['optim'])
            self.sched.load_state_dict(ckpt['sched'])

    def save(self, config, itr):
        if self.parallel:
            net_state = self.net.module.state_dict()
        else:
            net_state = self.net.state_dict()

        ckpt = {
            'config': config,
            'itr': itr,
            'net': net_state,
            'optim': self.optim.state_dict(),
            'sched': self.sched.state_dict(),
        }
        return ckpt

    def cuda(self, parallel=False):
        self.net.cuda()
        self.loss.cuda()
        if parallel:
            self.net = nn.DataParallel(self.net)
            self.loss.data_parallel()
            self.parallel = True

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def run(self, input_dict, h, w=None, mode='train', 
                nvs=True, ndc=True, pcd_size=None):
        for k in input_dict.keys():
            input_dict[k] = input_dict[k].cuda(non_blocking=True)

        if nvs:
            output_dict = self.net(input_dict, h, w, ndc, pcd_size)
            pred_rgb = output_dict['pred_rgb']                          # (bs, v, 3, h, w)
            bs, n_views = pred_rgb.shape[:2]
            pred_rgb = pred_rgb.flatten(0, 1)                           # (bs * v, 3, h, w)
            
            # find correspondences across views
            match = viz_mtx = None
            if 'uv' in output_dict:
                uv, viz = output_dict['uv'], output_dict['viz']         # (bs, v, p)
                # multi-view correspondences
                match = F.grid_sample(
                    pred_rgb, uv.flatten(0, 1).unsqueeze(-2), 
                    align_corners=False
                ).squeeze(-1).reshape(bs, n_views, 3, -1)               # (bs, v, 3, p)
                match = match.permute(0, 3, 1, 2).flatten(0, 1)         # (bs * p, v, 3)
                # pairwise visibility matrix
                viz_mtx = viz.unsqueeze(1) * viz.unsqueeze(2)           # (bs, v, v, p)
                viz_mtx = viz_mtx.permute(0, 3, 1, 2).flatten(0, 1)     # (bs * p, v, v)
                viz_mtx = viz_mtx * ~torch.eye( # ignore self-pairings
                    viz_mtx.size(-1), device=viz_mtx.device, 
                    dtype=torch.bool
                )

            if 'tgt_rgb' in input_dict:
                tgt_rgb = input_dict['tgt_rgb'].flatten(0, 1)
            else:
                tgt_rgb = output_dict['tgt_rgb'].flatten(0, 1)

            if tgt_rgb.shape[-2:] != pred_rgb.shape[-2:]:
                tgt_rgb = F.interpolate(
                    tgt_rgb, pred_rgb.shape[-2:], mode='bilinear', 
                    align_corners=False
                )

            style = input_dict['style']
            style = style.repeat_interleave(n_views, 0)
        else:
            output_dict = self.net(input_dict)
            pred_rgb = output_dict['pred_rgb']
            tgt_rgb = input_dict['src_rgb']
            style = input_dict['style']
            match = viz_mtx = None

        loss_dict = self.loss(pred_rgb, tgt_rgb, style, match, viz_mtx)
        loss = loss_dict['total']

        if mode == 'train':
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.sched.step()

        loss_dict.pop('total')
        for k in loss_dict.keys():
            loss_dict[k] = loss_dict[k].cpu()

        pred_rgb = torch.clamp(pred_rgb.detach(), 0, 1)
        output_dict = {
            'pred': pred_rgb.cpu(),
            'target': tgt_rgb.cpu(),
            'style': style.cpu(),
        }

        return output_dict, loss_dict


def make_optim_sched(config, params):
    if config['optim'] == 'sgd':
        optim = torch.optim.SGD(
            params, config['lr'], config['momentum'],
            weight_decay=config['weight_decay']
        )
    elif config['optim'] == 'adam':
        optim = torch.optim.Adam(
            params, config['lr'], (config['beta1'], config['beta2'])
        )
    elif config['optim'] == 'adamw':
        optim = torch.optim.AdamW(
            params, config['lr'], (config['beta1'], config['beta2'])
        )
    else:
        raise NotImplementedError(
            '[ERROR] invalid optimizer: {:s}'.format(config['optim'])
        )

    sched = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=config['decay_itrs'], gamma=config['decay']
    )

    return optim, sched


def make_inpainting_trainer(config):
    if config['3d']:
        netG = Model3D(config['encoder'], config['decoder'])
    else:
        netG = Model2D(config['encoder'], config['decoder'])
    netD = PatchDiscriminator(config['discriminator'])
    optimG, schedG = make_optim_sched(config['train'], netG.parameters())
    optimD, schedD = make_optim_sched(config['train'], netD.parameters())
    render_loss = InpaintingLoss(config['loss']['render'])
    D_GAN_loss = GANLoss(config['loss']['ganD'])
    G_GAN_loss = GANLoss(config['loss']['ganG'])

    trainer = InpaintingTrainer(
        netG, netD, optimG, optimD, schedG, schedD, 
        render_loss, D_GAN_loss, G_GAN_loss, config['loss']['weights']
    )
    trainer.cuda(parallel=config.get('_parallel'))
    return trainer


def make_stylization_trainer(config):
    if config['3d']:
        net = Model3D(config['encoder'], config['decoder'])
    else:
        net = Model2D(config['encoder'], config['decoder'])
    params = net.convert_for_stylization(
        config['stylizer'],
        freeze_enc=config.get('freeze_enc', True),
        freeze_dec=config.get('freeze_dec', True)
    )
    
    optim, sched = make_optim_sched(config['train'], params)
    loss = StylizationLoss(config['loss'])

    trainer = StylizationTrainer(net, optim, sched, loss)
    trainer.cuda(parallel=config.get('_parallel'))
    return trainer