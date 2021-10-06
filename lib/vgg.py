import torch
import torch.nn as nn


def make_vgg(pool='max'):
    if pool == 'max':
        Pool2d = nn.MaxPool2d
    elif pool == 'mean':
        Pool2d = nn.AvgPool2d
    else:
        raise NotImplementedError(
            '[ERROR] invalid pooling operator: {:s}'.format(pool)
        )
    return nn.Sequential(
        nn.Conv2d(3, 3, 1),     # input remapping [0, 1] -> [0, 255]
        nn.ReflectionPad2d(1), nn.Conv2d(3, 64, 3),
        nn.ReLU(inplace=True),  # relu1-1
        nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3),
        nn.ReLU(inplace=True),  # relu1-2
        Pool2d(2, 2),
        nn.ReflectionPad2d(1), nn.Conv2d(64, 128, 3),
        nn.ReLU(inplace=True),  # relu2-1
        nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3),
        nn.ReLU(inplace=True),  # relu2-2
        Pool2d(2, 2),
        nn.ReflectionPad2d(1), nn.Conv2d(128, 256, 3),
        nn.ReLU(inplace=True),  # relu3-1
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3),
        nn.ReLU(inplace=True),  # relu3-2
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3),
        nn.ReLU(inplace=True),  # relu3-3
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3),
        nn.ReLU(inplace=True),  # relu3-4
        Pool2d(2, 2),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 512, 3),
        nn.ReLU(inplace=True),  # relu4-1
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3),
        nn.ReLU(inplace=True),  # relu4-2
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3),
        nn.ReLU(inplace=True),  # relu4-3
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3),
        nn.ReLU(inplace=True),  # relu4-4
        Pool2d(2, 2),
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3),
        nn.ReLU(inplace=True),  # relu5-1
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3),
        nn.ReLU(inplace=True),  # relu5-2
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3),
        nn.ReLU(inplace=True),  # relu5-3
        nn.ReflectionPad2d(1), nn.Conv2d(512, 512, 3),
        nn.ReLU(inplace=True),  # relu5-4
    )


class NormalizedVGG(nn.Module):
    def __init__(self, layer=None, pool='max', pretrained=True):
        super(NormalizedVGG, self).__init__()

        vgg = make_vgg(pool)
        if pretrained:
            vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
            for param in vgg.parameters():
                param.requires_grad = False

        if layer is not None:
            assert layer in (1, 2, 3, 4, 5), \
                '[ERROR] VGG layer must be from 1 to 5, got {:d}'.format(layer)
        self.layer = layer

        self.slice1 = vgg[:4]    # input -> relu1_1
        self.slice2 = vgg[4:11]  # relu1_1 -> relu2_1
        self.slice3 = vgg[11:18] # relu2_1 -> relu3_1
        self.slice4 = vgg[18:31] # relu3_1 -> relu4_1
        self.slice5 = vgg[31:44] # relu4_1 -> relu5_1

    def forward(self, x):
        relu1 = self.slice1(x)
        if self.layer == 1:
            return relu1
        relu2 = self.slice2(relu1)
        if self.layer == 2:
            return relu2
        relu3 = self.slice3(relu2)
        if self.layer == 3:
            return relu3
        relu4 = self.slice4(relu3)
        if self.layer == 4:
            return relu4
        relu5 = self.slice5(relu4)
        if self.layer == 5:
            return relu5
        return [relu1, relu2, relu3, relu4, relu5]


# VGG decoder for inverting relu3_1
def make_dvgg3(in_dim=256, up='nearest'):
    if up == 'shuffle':
        return nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(in_dim, 512, 3),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(64, 3, 3),
        )
    if up == 'nearest':
        Upsampling2d = nn.UpsamplingNearest2d
    elif up == 'bilinear':
        Upsampling2d = nn.UpsamplingBilinear2d
    else:
        raise NotImplementedError(
            '[ERROR] invalid upsampling operator: {:s}'.format(up)
        )
    return nn.Sequential(
        nn.ReflectionPad2d(1), nn.Conv2d(in_dim, 128, 3),
        nn.ReLU(inplace=True),
        Upsampling2d(scale_factor=2),
        nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1), nn.Conv2d(128, 64, 3),
        nn.ReLU(inplace=True),
        Upsampling2d(scale_factor=2),
        nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1), nn.Conv2d(64, 3, 3),
    )


# VGG decoder for inverting relu4_1
def make_dvgg4(in_dim=512, up='nearest'):
    if up == 'shuffle':
        return nn.Sequential(
        nn.ReflectionPad2d(1), nn.Conv2d(in_dim, 1024, 3),
        nn.ReLU(inplace=True),
        nn.PixelShuffle(2),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3),
        nn.ReLU(inplace=True),
        *make_dvgg3(256, up),
    )
    if up == 'nearest':
        Upsampling2d = nn.UpsamplingNearest2d
    elif up == 'bilinear':
        Upsampling2d = nn.UpsamplingBilinear2d
    else:
        raise NotImplementedError(
            '[ERROR] invalid upsampling operator: {:s}'.format(up)
        )
    return nn.Sequential(
        nn.ReflectionPad2d(1), nn.Conv2d(in_dim, 256, 3),
        nn.ReLU(inplace=True),
        Upsampling2d(scale_factor=2),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3),
        nn.ReLU(inplace=True),
        *make_dvgg3(256, up),
    )


def make_dvgg(layer, in_dim, up='nearest', pretrained=False):
    if layer == 3:
        dvgg = make_dvgg3(in_dim, up)
        if pretrained:
            assert in_dim == 256, \
                ('[ERROR] pre-trained model accepts 256-dimensional features, '
                 'got {:d}'.format(in_dim)
                )
            dvgg.load_state_dict(torch.load('models/dvgg3_normalised.pth'))

    elif layer == 4:
        dvgg = make_dvgg4(in_dim, up)
        if pretrained:
            assert in_dim == 512, \
                ('[ERROR] pre-trained model accepts 512-dimensional features, '
                 'got {:d}'.format(in_dim)
                )
            dvgg.load_state_dict(torch.load('models/dvgg4_normalised.pth'))
    else:
        raise NotImplementedError(
            '[ERROR] invalid VGG layer: {:d}'.format(layer)
        )

    if pretrained:
        for param in dvgg.parameters():
            param.requires_grad = False
    return dvgg