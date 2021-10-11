import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import AdaIN, AdaAttN
from .vgg import NormalizedVGG


class PixelLoss(nn.Module):
    """ pixel-wise loss """

    def __init__(self, loss_type='l1', reduction='mean'):
        super(PixelLoss, self).__init__()

        if loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss(reduction=reduction)
        elif loss_type == 'huber':
            self.criterion = nn.HuberLoss(reduction=reduction)
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError(
                '[ERROR] invalid loss type: {:s}'.format(loss_type)
            )

        self.reduction = reduction

    def forward(self, pred, target=None):
        if target is None:
            target = torch.zeros_like(pred)
        loss = self.criterion(pred, target)
        if self.reduction == 'sum':
            loss /= len(pred)
        return loss


class MatchingLoss(nn.Module):
    """ correspondence matching loss """

    def __init__(self, loss_type='l1'):
        super(MatchingLoss, self).__init__()

        self.criterion = PixelLoss(loss_type, reduction='none')

    def forward(self, match, viz_mtx=None):
        """
        Args:
            match (float tensor, (p, v, 3)): candidate matches.
            viz_mtx (bool tensor, (p, v, v)): pairwise visibility matrix.

        Returns:
            loss (float): average loss over all visible corrspondences.
        """
        tmp1, tmp2 = match.unsqueeze(1), match.unsqueeze(2)
        loss = self.criterion(tmp1, tmp2).sum(-1)       # (p, v, v)
        if viz_mtx is not None:
            loss = (loss * viz_mtx).sum() / viz_mtx.sum()
        else:
            loss = loss.mean()
        return loss


class DistillLoss(nn.Module):
    """ feature distillation loss """

    def __init__(self, loss_type='l1', layer=3, norm=None,
                 reduction='mean'):
        super(DistillLoss, self).__init__()

        self.criterion = PixelLoss(loss_type, reduction=reduction)

        assert layer in (1, 2, 3, 4, 5), \
            '[ERROR] invalid VGG layer: {:s}'.format(layer)
        self.layer = layer

        if norm is not None:
            assert norm in ('mean', 'instance'), \
                '[ERROR] invalid feature normalization: {:s}'.format(norm)
        self.norm = norm

    def forward(self, student_feats, teacher_feats):
        sf, tf = student_feats, teacher_feats[self.layer - 1]
        if sf.shape != tf.shape:
            tf = F.interpolate(
                tf, size=sf.shape[-2:], mode='bilinear', 
                align_corners=False
            )
        if self.norm == 'mean':
            sf = sf - sf.mean((-2, -1), keepdim=True)
            tf = tf - sf.mean((-2, -1), keepdim=True)
        elif self.norm == 'instance':
            sf = F.instance_norm(sf)
            tf = F.instance_norm(tf)
        loss = self.criterion(sf, tf)
        return loss


class VGGContentLoss(nn.Module):
    """ VGG content loss """

    def __init__(self, loss_type='l1', layers=[3], norm=None, 
                 reduction='mean'):
        super(VGGContentLoss, self).__init__()
        
        self.criterion = PixelLoss(loss_type, reduction=reduction)

        for l in layers:
            assert l in (1, 2, 3, 4, 5), \
                '[ERROR] invalid VGG layer: {:s}'.format(l)
        self.layers = layers

        self.adain = self.adaattn = None
        if norm is not None:
            if norm == 'adain':
                self.adain = AdaIN()
            elif norm == 'adaattn':
                self.adaattn = AdaAttN()
            else:
                raise NotImplementedError(
                    '[ERROR] invalid content normalization: {:s}'.format(norm)
                )
        self.norm = norm

    def forward(self, pred_feats, content_feats, style_feats=None):
        if self.norm is not None:
            assert style_feats is not None, \
                '[ERROR] style features must be given for AdaAttN evaluation'
            if self.norm == 'adaattn':
                q, k = content_feats[0], style_feats[0]
        
        loss = 0
        for i in range(len(pred_feats)):
            p, c = pred_feats[i], content_feats[i]

            # accumulate query and key features for AdaAttN
            if self.norm == 'adaattn' and i > 0:
                s = style_feats[i]
                q = F.interpolate(
                    q, size=c.shape[-2:], mode='bilinear', align_corners=False
                )
                k = F.interpolate(
                    k, size=s.shape[-2:], mode='bilinear', align_corners=False
                )
                q = torch.cat([q, c], 1)
                k = torch.cat([k, s], 1)
            
            if i + 1 in self.layers:
                if self.norm == 'adain':
                    c = self.adain(c, style_feats[i])
                if self.norm == 'adaattn':
                    c = self.adaattn(q, k, c, style_feats[i])
                loss += self.criterion(p, c)
        return loss


class VGGStyleLoss(nn.Module):
    """ VGG style loss """

    def __init__(self, loss_type='mse', layers=[1, 2, 3], 
                 stats=['mean', 'gram'], reduction='sum'):
        super(VGGStyleLoss, self).__init__()

        self.criterion = PixelLoss(loss_type, reduction=reduction)

        for l in layers:
            assert l in (1, 2, 3, 4, 5), \
                '[ERROR] invalid VGG layer: {:s}'.format(l)
        self.layers = layers

        for s in stats:
            assert s in ('mean', 'std', 'gram'), \
                '[ERROR] invalid style statistic: {:s}'.format(s)
        self.stats = stats

    def _gram(self, x):
        bs, c, h, w = x.size()
        x = x.view(bs, c, h * w)
        gram = torch.bmm(x, x.transpose(2, 1)) / (c * h * w)
        return gram

    def forward(self, pred_feats, style_feats):
        loss = 0
        for l in self.layers:
            p, s = pred_feats[l - 1], style_feats[l - 1]
            if 'mean' in self.stats:
                loss += self.criterion(p.mean((-2, -1)), s.mean((-2, -1)))
            if 'std' in self.stats:
                loss += self.criterion(p.std((-2, -1)), s.std((-2, -1)))
            if 'gram' in self.stats:
                loss += self.criterion(self._gram(p), self._gram(s))
        return loss


class _BaseLoss(nn.Module):

    def __init__(self):
        super(_BaseLoss, self).__init__()

        self.vgg = None
        self.criteria = nn.ModuleList()

    def data_parallel(self):
        if self.vgg is not None:
            self.vgg = nn.DataParallel(self.vgg)


class InpaintingLoss(_BaseLoss):

    def __init__(self, cfg):
        super(InpaintingLoss, self).__init__()

        self.terms = cfg['terms']
        self.weights = cfg['weights']

        if 'content' in self.terms or 'distill' in self.terms:
            self.vgg = NormalizedVGG()

        for i in range(len(self.terms)):
            if self.terms[i] == 'pixel':
                self.criteria.append(PixelLoss(**cfg['pixel']))
            elif self.terms[i] == 'content':
                self.criteria.append(VGGContentLoss(**cfg['content']))
            elif self.terms[i] == 'distill':
                self.criteria.append(DistillLoss(**cfg['distill']))
            elif self.terms[i] == 'match':
                self.criteria.append(MatchingLoss(**cfg['match']))
            else:
                raise NotImplementedError(
                    '[ERROR] invalid loss term: {:s}'.format(self.terms[i])
                )
            
    def forward(self, pred, target, pyramid, match=None, viz_mtx=None):
        if self.vgg is not None:
            with torch.no_grad():
                target_feats = self.vgg(target)

        loss_dict = {'total': 0}
        for i in range(len(self.criteria)):
            if self.terms[i] == 'pixel':
                loss = self.criteria[i](pred, target)
            elif self.terms[i] == 'content':
                pred_feats = self.vgg(pred)
                loss = self.criteria[i](pred_feats, target_feats)
            elif self.terms[i] == 'distill':
                loss = self.criteria[i](pyramid, target_feats)
            elif self.terms[i] == 'match':
                assert match is not None, '[ERROR] matches do not exist'
                loss = self.criteria[i](match, viz_mtx)
            loss_dict[self.terms[i]] = loss
            loss_dict['total'] += self.weights[i] * loss
        return loss_dict


class StylizationLoss(_BaseLoss):

    def __init__(self, cfg):
        super(StylizationLoss, self).__init__()

        self.terms = cfg['terms']
        self.weights = cfg['weights']

        if 'content' in self.terms or 'style' in self.terms:
            self.vgg = NormalizedVGG()
        
        for i in range(len(self.terms)):
            if self.terms[i] == 'content':
                self.criteria.append(VGGContentLoss(**cfg['content']))
            elif self.terms[i] == 'style':
                self.criteria.append(VGGStyleLoss(**cfg['style']))
            elif self.terms[i] == 'match':
                self.criteria.append(MatchingLoss(**cfg['match']))
            else:
                raise NotImplementedError(
                    '[ERROR] invalid loss term: {:s}'.format(self.terms[i])
                )

    def forward(self, pred, content, style, match=None, viz_mtx=None):
        if self.vgg is not None:
            pred_feats = self.vgg(pred)
            with torch.no_grad():
                content_feats = self.vgg(content)
                style_feats = self.vgg(style)

        loss_dict = {'total': 0}
        for i in range(len(self.criteria)):
            if self.terms[i] == 'content':
                loss = self.criteria[i](pred_feats, content_feats, style_feats)
            elif self.terms[i] == 'style':
                loss = self.criteria[i](pred_feats, style_feats)
            elif self.terms[i] == 'match':
                assert match is not None, '[ERROR] matches do not exist'
                loss = self.criteria[i](match, viz_mtx)
            loss_dict[self.terms[i]] = loss
            loss_dict['total'] += self.weights[i] * loss
        return loss_dict


class GANLoss(nn.Module):

    def __init__(self, loss_type='hinge'):
        super(GANLoss, self).__init__()

        assert loss_type in ['nsgan', 'lsgan', 'wgan', 'hinge'], \
            'ERROR: invalid GAN loss type: {:s}'.format(loss_type)
        self.loss_type = loss_type

        self.register_buffer('real_label', torch.Tensor([1]))
        self.register_buffer('fake_label', torch.Tensor([0]))

        if loss_type == 'nsgan':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.criterion = nn.MSELoss()

    def _forward(self, x, is_real, is_critic=True):
        if self.loss_type == 'hinge':
            if is_critic:
                if is_real:
                    x = -x
                return F.relu(1 + x).mean()
            else:
                return (-x).mean()
        elif self.loss_type == 'wgan':
            if is_real:
                return (-x).mean()
            else:
                return x.mean()
        else:
            y = (self.real_label if is_real else self.fake_label).expand_as(x)
            return self.criterion(x, y)

    def forward(self, x, is_real, is_critic=True):
        if not isinstance(x, (list, tuple)):
            x = [x]
        loss = self._forward(x[-1], is_real, is_critic)
        return loss