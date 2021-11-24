import torch
import torch.nn as nn
import torch.nn.functional as F

import lpips


class RMSE(nn.Module):
    """ Root mean square error """

    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, im1, im2, mask=None):
        assert im1.shape == im2.shape, 'input shape mismatch'

        rmse = (im1 - im2).pow(2)
        if mask is not None:
            rmse = (rmse * mask).flatten(1).sum(-1)
            rmse = rmse / mask.flatten(1).sum(-1)
        rmse = rmse.mean().sqrt()
        return rmse


class PSNR(nn.Module):
    """ Peak signal-to-noise ratio """

    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, im1, im2):
        assert im1.shape == im2.shape, 'input shape mismatch'

        bs = im1.size(0)
        mse = (im1 - im2).pow(2).mean() + 1e-8
        psnr = -10 * mse.log10()
        return psnr


class SSIM(nn.Module):
    """ Structural similarity index measure """
    
    def __init__(self, n_channels=3, kernel_size=11, sigma=1.5):
        super(SSIM, self).__init__()

        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.sigma = sigma

        tics = torch.arange(kernel_size)[:, None]
        kernel = torch.exp(
            -(tics - kernel_size // 2).pow(2) / (2 * sigma ** 2)
        )
        kernel = kernel / kernel.sum()
        kernel = torch.mm(kernel, kernel.t())
        kernel = kernel.expand(n_channels, 1, -1, -1).contiguous()
        self.register_buffer('kernel', kernel)

        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def forward(self, im1, im2):
        assert im1.shape == im2.shape, 'input shape mismatch'
        assert im1.size(1) == self.n_channels, 'number of channels mismatch'

        mu1 = F.conv2d(im1, self.kernel, groups=self.n_channels)
        mu2 = F.conv2d(im2, self.kernel, groups=self.n_channels)
        
        im1_sq, im2_sq = im1.pow(2), im2.pow(2)
        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        im1_im2, mu1_mu2 = im1 * im2, mu1 * mu2

        sigma1_sq = F.conv2d(im1_sq, self.kernel, groups=self.n_channels) - mu1_sq
        sigma2_sq = F.conv2d(im2_sq, self.kernel, groups=self.n_channels) - mu2_sq
        sigma12 = F.conv2d(im1_im2, self.kernel, groups=self.n_channels) - mu1_mu2

        tmp1 = (2 * mu1_mu2 + self.c1) * (2 * sigma12 + self.c2)
        tmp2 = (mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2)
        ssim_map = tmp1 / tmp2
        ssim = ssim_map.mean()
        return ssim


class LPIPS(nn.Module):
    """ Learned perceptial image patch similarity """

    def __init__(self, net='alex', calibrate=True):
        super(LPIPS, self).__init__()

        assert net in ('alex', 'vgg'), \
            '[ERROR] invalid base network for LPIPS: {:s}'.format(net)

        self.lpips = lpips.LPIPS(net=net, lpips=calibrate, spatial=True)

    def forward(self, im1, im2, mask=None):
        ## NOTE: input images are in [0, 1],
        ## we set normalize=True to rescale input to [-1, 1]
        pips = self.lpips(im1, im2, normalize=True)     # (bs, 1, h, w)
        if mask is not None:
            pips = (pips * mask).flatten(1).sum(-1)
            pips = pips / mask.flatten(1).sum(-1)
        pips = pips.mean()
        return pips