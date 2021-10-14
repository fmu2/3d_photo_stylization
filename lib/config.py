import yaml


INPAINT2D_DEFAULTS = {
    '3d': False,

    'encoder': {
        'arch': 'vgg',

        'vgg': {
            'layer': 4,
            'pool': 'max',
        },
    },

    'decoder': {
        'arch': 'vgg',

        'vgg': {
            'layer': 4,
            'in_dim': 512,
            'up': 'nearest',
            'pretrained': False,

            'out_norm': False,
        },
    },

    'discriminator': {
        'spectral_norm': True,
    },

    'loss': {
        'render': {
            'terms': ['pixel', 'content'],
            'weights': [1, 1],

            'pixel': {
                'loss_type': 'l1',
                'reduction': 'mean',
            },

            'content': {
                'loss_type': 'l1',
                'layers': [4],
                'norm': None,
                'reduction': 'mean',
            },
        },

        'ganD': 'hinge',
        'ganG': 'hinge',

        'weights': [1, 0, 0],
    },

    'data': {
        'name': 'coco',
        'im_size': 256,
    },

    'train': {
        'batch_size': 8,
        'n_itrs': 50000,

        'optim': 'adam',
        'lr': 1e-4,
        'beta1': 0.9,
        'beta2': 0.999,
        'decay_itrs': [-1],
        'decay': 0.5,
    },
}


INPAINT3D_DEFAULTS = {
    '3d': True,
    'ndc': True,
    
    'encoder': {
        'arch': 'pcd',

        'pcd': {
            'n_levels': 2,
            
            'aggregate': 'max',
            'norm': 'batch',
            'actv': 'relu',
            'res': True,

            'in_conv': 'mr',
            'in_dim': 64,
            'in_radius': 0.015,
            'in_k': 16,

            'block': 'mr',
            'scale_factor': [4, 4],
            'depth': [1, 1],
            'dims': [128, 256],
            'radius': [[[0.015, 0.025]], [[0.025, 0.05]]],
            'k': [[[16, 16]], [[16, 16]]],

            'out_relu': False,
            'up': 'linear',
        },
    },

    'decoder': {
        'arch': 'unet',

        'unet': {
            'n_levels': 2,

            'in_dim': 256,
            'down': 'conv',
            'down_actv': 'leaky_relu',
            'up': 'conv',
            'up_actv': 'conv',
            'norm': None,
        },
    },

    'discriminator': {
        'spectral_norm': True,
    },

    'loss': {
        'render': {
            'terms': ['pixel', 'content', 'match'],
            'weights': [1, 1, 1],

            'pixel': {
                'loss_type': 'l1',
                'reduction': 'mean',
            },

            'content': {
                'loss_type': 'l1',
                'layers': [3],
                'norm': None,
                'reduction': 'mean',
            },

            'match': {
                'loss_type': 'l1',
            },
        },

        'ganD': 'hinge',
        'ganG': 'hinge',

        'weights': [1, 0, 0],
    },

    'data': {
        'name': 'pcd',
        'n_target_views': 2,
        'x_lim': [-0.05, 0.05],
        'y_lim': [-0.05, 0.05],
        'z_lim': [-0.05, 0.15],
    },

    'train': {
        'batch_size': 8,
        'n_itrs': 50000,

        'h': 224,
        'w': 224,
        'pcd_size': 262144,

        'optim': 'adam',
        'lr': 1e-4,
        'beta1': 0.9,
        'beta2': 0.999,
        'decay_itrs': [-1],
        'decay': 0.5,
    },
}


STYLIZE2D_DEFAULTS = {
    '3d': False,

    'encoder': {
        'arch': 'vgg',

        'vgg': {
            'layer': 4,
            'pool': 'max',
        },
    },

    'decoder': {
        'arch': 'vgg',

        'vgg': {
            'layer': 4,
            'in_dim': 512,
            'up': 'nearest',
            'pretrained': True,

            'out_norm': False,
        },
    },

    'stylizer': {
        'arch': 'linear',

        'linear': {
            'vgg_layer': 4,
            'vgg_pool': 'max',
            'in_dim': 512,
            'n_embed_layers': 3,
        },
    },

    'loss': {
        'terms': ['content', 'style'],
        'weights': [1, 0.02],

        'content': {
            'loss_type': 'mse',
            'layers': [4],
            'norm': None,
            'reduction': 'mean',
        },

        'style': {
            'loss_type': 'mse',
            'layers': [1, 2, 3, 4],
            'stats': ['mean', 'gram'],
            'reduction': 'sum',
        },
    },

    'data': {
        'name': 'coco',
        'im_size': 256,
    },

    'style': {
        'name': 'wikiart',
        'im_size': 256,
    },

    'train': {
        'batch_size': 8,
        'n_itrs': 10000,

        'optim': 'adam',
        'lr': 1e-4,
        'beta1': 0.9,
        'beta2': 0.999,
        'decay_itrs': [-1],
        'decay': 0.5,
    },
}


STYLIZE3D_DEFAULTS = {
    '3d': True,
    'ndc': True,

    'stylizer': {
        'arch': 'adain',

        'adain': {
            'vgg_layer': 3,
            'vgg_pool': 'mean',
            'n_zip_layers': 2,
        },
    },

    'loss': {
        'terms': ['content', 'style', 'match'],
        'weights': [1, 0.02, 1],

        'content': {
            'loss_type': 'mse',
            'layers': [3],
            'norm': None,
            'reduction': 'mean',
        },

        'style': {
            'loss_type': 'mse',
            'layers': [1, 2, 3],
            'stats': ['mean', 'gram'],
            'reduction': 'sum',
        },

        'match': {
            'loss_type': 'l1',
        },
    },

    'data': {
        'name': 'pcd',
        'n_target_views': 2,
        'x_lim': [-0.05, 0.05],
        'y_lim': [-0.05, 0.05],
        'z_lim': [-0.05, 0.15],
    },

    'style': {
        'name': 'wikiart',
        'im_size': 256,
    },

    'train': {
        'batch_size': 8,
        'n_itrs': 10000,

        'h': 224,
        'w': 224,
        'pcd_size': 112896,

        'optim': 'adam',
        'lr': 1e-4,
        'beta1': 0.9,
        'beta2': 0.999,
        'decay_itrs': [-1],
        'decay': 0.5,
    },
}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_file, mode='inpaint'):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if mode == 'inpaint':
        if config['3d']:
            defaults = INPAINT3D_DEFAULTS
        else:
            defaults = INPAINT2D_DEFAULTS
    elif mode == 'stylize':
        if config['3d']:
            defaults = STYLIZE3D_DEFAULTS
        else:
            defaults = STYLIZE2D_DEFAULTS
    else:
        raise ValueError('invalid config mode: {}'.format(mode))
    _merge(defaults, config)
    return config