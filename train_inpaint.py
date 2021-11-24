import os
import random
import argparse

import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from lib.config import load_config
from lib.data import make_dataset, collate_fn, cycle
from lib.trainer import make_inpainting_trainer
from lib.util import *


def main(config):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    set_log_path(ckpt_path)
    writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))

    ###########################################################################
    """ dataset """

    train_set = make_dataset(
        config['data'], args.data_dir, split='train', nvs=config['3d']
    )
    train_loader = DataLoader(
        train_set, batch_size=config['train']['batch_size'],
        collate_fn=collate_fn, num_workers=8, shuffle=True, drop_last=True
    )
    train_iterator = cycle(train_loader)

    print('train data size: {:d}'.format(len(train_set)))

    val_set = make_dataset(
        config['data'], args.data_dir, split='val', nvs=config['3d']
    )
    val_loader = DataLoader(
        val_set, batch_size=config['train']['batch_size'],
        collate_fn=collate_fn, num_workers=8, shuffle=False, drop_last=True
    )

    print('val data size: {:d}'.format(len(val_set)))

    ###########################################################################
    """ trainer """

    n_itrs = config['train']['n_itrs']
    itr0 = 0
    if config.get('_resume'):
        ckpt_name = os.path.join(ckpt_path, 'inpaint-last.pth')
        try:
            check_file(ckpt_name)
            ckpt = torch.load(ckpt_name)
            trainer = make_inpainting_trainer(ckpt['config'])
            trainer.load(ckpt)
            itr0 = ckpt['itr']
            print('checkpoint loaded, train from itr {:d}'.format(itr0))
        except:
            config.pop('_resume')
            trainer = make_inpainting_trainer(config)
            itr0 = 0
            print('WARNING: checkpoint loading failed, train from scratch')
    else:
        trainer = make_inpainting_trainer(config)
        print('trainer initialized, train from scratch')
        yaml.dump(
            config, open(os.path.join(ckpt_path, 'inpaint-config.yaml'), 'w')
        )

    if itr0 == 0:
        ckpt = trainer.save(config, 0)
        torch.save(ckpt, os.path.join(ckpt_path, 'inpaint-init.pth'))
        print('initial inpainting model saved')

    ###########################################################################
    """ train & val """
    
    loss_list = ['pixel', 'content', 'distill', 'match', 'ganG', 'ganD']
    train_losses = {k: AverageMeter() for k in loss_list}
    val_losses = {k: AverageMeter() for k in loss_list}

    metric_list = ['rmse', 'psnr', 'ssim']
    train_metrics = {k: AverageMeter() for k in metric_list}
    val_metrics = {k: AverageMeter() for k in metric_list}

    timer = Timer()

    for itr in range(itr0 + 1, n_itrs + 1):
        trainer.train()

        input_dict = next(train_iterator)
        out_dict, loss_dict, metric_dict = trainer.run(
            input_dict=input_dict, 
            h=config['train']['h'], 
            w=config['train'].get('w'),
            mode='train', 
            nvs=config['3d'], 
            ndc=config.get('ndc', True),
            pcd_size=config['train'].get('pcd_size')
        )
        
        for k in loss_dict.keys():
            train_losses[k].update(loss_dict[k].item())
            writer.add_scalars(k, {
                'train_inpaint': train_losses[k].item()}, itr
            )
        
        for k in metric_dict.keys():
            train_metrics[k].update(metric_dict[k].item())
            writer.add_scalars(
                k, {'train_inpaint': train_metrics[k].item()}, itr
            )

        if itr % args.print_freq == 0 or itr == 1:
            t_elapsed = time_str(timer.end())
            log_str = '[{:04d}/{:04d}] '.format(
                itr // args.print_freq, n_itrs // args.print_freq
            )
            for k in loss_dict.keys():
                log_str += '{:s} {:.3f} | '.format(k, train_losses[k].item())
            for k in metric_dict.keys():
                log_str += '{:s} {:.3f} | '.format(k, train_metrics[k].item())
            log_str += t_elapsed
            log(log_str, 'inpaint-log.txt')

            for k in out_dict.keys():
                writer.add_images(
                    tag='train/inpaint/{:04d}/{:s}'.format(
                        itr // args.print_freq, k
                    ), 
                    img_tensor=out_dict[k], 
                    global_step=itr // args.print_freq
                )

            writer.flush()
            for k in loss_list:
                train_losses[k].reset()
            for k in metric_list:
                train_metrics[k].reset()

            ckpt = trainer.save(config, itr)
            torch.save(ckpt, os.path.join(ckpt_path, 'inpaint-last.pth'))
            timer.start()

        if itr % args.val_freq == 0:
            trainer.eval()

            for input_dict in val_loader:
                with torch.no_grad():
                    out_dict, loss_dict, metric_dict = trainer.run(
                        input_dict=input_dict, 
                        h=config['train']['h'], 
                        w=config['train'].get('w'),
                        mode='val', 
                        nvs=config['3d'],
                        ndc=config.get('ndc', True), 
                        pcd_size=config['train'].get('pcd_size')
                    )

                for k in loss_dict.keys():
                    val_losses[k].update(loss_dict[k].item())
                
                for k in metric_dict.keys():
                    val_metrics[k].update(metric_dict[k].item())

            for k in loss_dict.keys():
                writer.add_scalars(
                    k, {'val_inpaint': val_losses[k].item()}, itr
                )
            
            for k in metric_dict.keys():
                writer.add_scalars(
                    k, {'val_inpaint': val_metrics[k].item()}, itr
                )
                    
            t_elapsed = time_str(timer.end())
            log_str = '[{:04d}/{:04d} val] '.format(
                itr // args.print_freq, n_itrs // args.print_freq
            )
            for k in loss_dict.keys():
                log_str += '{:s} {:.3f} | '.format(k, val_losses[k].item())
            for k in metric_dict.keys():
                log_str += '{:s} {:.3f} | '.format(k, val_metrics[k].item())
            log_str += t_elapsed + '\n'
            log(log_str, 'inpaint-log.txt')

            for k in out_dict.keys():
                writer.add_images(
                    tag='val/inpaint/{:04d}/{:s}'.format(
                        itr // args.print_freq, k
                    ), 
                    img_tensor=out_dict[k], 
                    global_step=itr // args.print_freq
                )

            writer.flush()
            for k in loss_list:
                val_losses[k].reset()
            for k in metric_list:
                val_metrics[k].reset()
            timer.start()

    ###########################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, 
                        help='data directory')
    parser.add_argument('-c', '--config', type=str, 
                        help='config file path')
    parser.add_argument('-n', '--name', type=str, default='inpaint',
                        help='job name')
    parser.add_argument('-g', '--gpu', type=str, default='0', 
                        help='GPU device IDs')
    parser.add_argument('-pf', '--print_freq', type=int, default=1, 
                        help='print frequency (x100 itrs)')
    parser.add_argument('-vf', '--val_freq', type=int, default=100,
                        help='validation frequency (x100 itrs)')
    args = parser.parse_args()

    args.print_freq *= 100
    args.val_freq *= 100

    # set up checkpoint folder
    if not os.path.exists('log'):
        os.makedirs('log')
    ckpt_path = os.path.join('log', args.name)
    ensure_path(ckpt_path)

    # load config
    try:
        config_path = os.path.join(ckpt_path, 'inpaint-config.yaml')
        check_file(config_path)
        config = load_config(config_path, mode='inpaint')
        print('config loaded from checkpoint folder')
        config['_resume'] = True
    except:
        check_file(args.config)
        config = load_config(args.config, mode='inpaint')
        print('config loaded from command line')

    # configure GPUs
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu
    set_gpu(args.gpu)

    main(config)