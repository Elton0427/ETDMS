import torch
import data as Data
import model as Model
import argparse
import cv2
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np


def train_stage1(opt, logger, tb_logger, wandb_logger=None):
    # Stage 1: SwinIR Denoising
    logger.info('Training Stage 1: SwinIR Denoising')
    
    # Dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = Data.create_dataset2(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset2(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
    
    # Model
    stage1_opt = opt.copy()
    if 'stage1' in opt:
        stage1_opt['model'] = opt['stage1']
        # Ensure the model type is correctly set
        if stage1_opt['model']['which_model_G'] == 'swinir':
            # Use SwinIR model
            denoiser = Model.create_model(stage1_opt)
        else:
            # Handle error or use appropriate model
            logger.error("Unsupported model type for stage1")
    else:
        denoiser = Model.create_model(stage1_opt)
    
    # Training loop for Stage 1
    current_step = denoiser.begin_step
    current_epoch = denoiser.begin_epoch
    n_iter = opt['train']['n_iter']
    
    while current_step < n_iter:
        current_epoch += 1
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break
            denoiser.feed_data(train_data)
            denoiser.optimize_parameters()
            
            # Logging
            if current_step % opt['train']['print_freq'] == 0:
                logs = denoiser.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}> '.format(current_epoch, current_step)
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    tb_logger.add_scalar(k, v, current_step)
                logger.info(message)
            
            # Validation
            if current_step % opt['train']['val_freq'] == 0:
                avg_psnr = 0.0
                idx = 0
                result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                os.makedirs(result_path, exist_ok=True)
                
                if 'stage1' in opt and 'beta_schedule' in opt['stage1'] and 'val' in opt['stage1']['beta_schedule']:
                    denoiser.set_new_noise_schedule(opt['stage1']['beta_schedule']['val'], schedule_phase='val')
                for _, val_data in enumerate(val_loader):
                    idx += 1
                    denoiser.feed_data(val_data)
                    denoiser.test(continous=False)
                    visuals = denoiser.get_current_visuals()
                    
                    sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                    hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                    lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                    
                    # Save images
                    Metrics.save_img(hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                    Metrics.save_img(sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                    Metrics.save_img(lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                    
                    # Calculate metrics
                    # Ensure both images have the same shape
                    if sr_img.shape != hr_img.shape:
                        # If one is grayscale and other is RGB, convert to same format
                        if len(sr_img.shape) == 2 and len(hr_img.shape) == 3:
                            sr_img = np.stack([sr_img] * 3, axis=-1)
                        elif len(hr_img.shape) == 2 and len(sr_img.shape) == 3:
                            hr_img = np.stack([hr_img] * 3, axis=-1)
                        
                        # Resize if dimensions don't match
                        if sr_img.shape[:2] != hr_img.shape[:2]:
                            sr_img = cv2.resize(sr_img, (hr_img.shape[1], hr_img.shape[0]))
                    
                    # Calculate metrics
                    # Fix shape mismatch between sr_img and hr_img
                    print(f"SR shape: {sr_img.shape}, HR shape: {hr_img.shape}")  # Debug info
                    
                    if sr_img.shape != hr_img.shape:
                        # If one is grayscale (2D) and other is RGB (3D), convert grayscale to RGB
                        if len(sr_img.shape) == 2 and len(hr_img.shape) == 3:
                            sr_img = np.stack([sr_img] * 3, axis=-1)
                            print(f"Converted SR to RGB: {sr_img.shape}")
                        elif len(hr_img.shape) == 2 and len(sr_img.shape) == 3:
                            hr_img = np.stack([hr_img] * 3, axis=-1)
                            print(f"Converted HR to RGB: {hr_img.shape}")
                        
                        # If dimensions still don't match, resize to match
                        if sr_img.shape[:2] != hr_img.shape[:2]:
                            sr_img = cv2.resize(sr_img, (hr_img.shape[1], hr_img.shape[0]))
                            print(f"Resized SR to match HR: {sr_img.shape}")
                    
                    avg_psnr += Metrics.calculate_psnr(sr_img, hr_img)
                
                avg_psnr = avg_psnr / idx
                
                # Check if beta_schedule exists before calling set_new_noise_schedule
                if 'beta_schedule' in opt.get('stage1', {}) and 'train' in opt['stage1']['beta_schedule']:
                    denoiser.set_new_noise_schedule(opt['stage1']['beta_schedule']['train'], schedule_phase='train')
                
                # Log validation results
                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(current_epoch, current_step, avg_psnr))
                tb_logger.add_scalar('psnr', avg_psnr, current_step)
                
            # Save checkpoint
            if current_step % opt['train']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                denoiser.save_network(current_epoch, current_step)


def train_stage2(opt, logger, tb_logger, wandb_logger):
    logger.info('Training Stage 2: CLSR Super-Resolution')
    
    # Dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')
    
    # Model
    stage2_opt = opt.copy()
    stage2_opt['model'] = opt['model']
    diffusion = Model.create_model(stage2_opt)
    
    # Training loop for Stage 2
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
    
    while current_step < n_iter:
        current_epoch += 1
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            
            # Logging
            if current_step % opt['train']['print_freq'] == 0:
                logs = diffusion.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}> '.format(current_epoch, current_step)
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    tb_logger.add_scalar(k, v, current_step)
                logger.info(message)
            
            # Validation
            if current_step % opt['train']['val_freq'] == 0:
                avg_psnr = 0.0
                idx = 0
                result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                os.makedirs(result_path, exist_ok=True)
                
                diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
                for _, val_data in enumerate(val_loader):
                    idx += 1
                    diffusion.feed_data(val_data)
                    diffusion.test(continous=False)
                    visuals = diffusion.get_current_visuals()
                    
                    sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                    hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                    lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                    
                    # Save images
                    Metrics.save_img(hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                    Metrics.save_img(sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                    Metrics.save_img(lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                    
                    # Convert to same number of channels
                    if len(sr_img.shape) == 2 and len(hr_img.shape) == 3:
                        # SR is grayscale, HR is RGB - convert SR to RGB
                        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_GRAY2RGB)
                    elif len(sr_img.shape) == 3 and len(hr_img.shape) == 2:
                        # SR is RGB, HR is grayscale - convert HR to RGB
                        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_GRAY2RGB)
                    elif len(sr_img.shape) == 3 and len(hr_img.shape) == 3:
                        # Both are RGB, ensure same number of channels
                        if sr_img.shape[2] != hr_img.shape[2]:
                            if sr_img.shape[2] == 1:
                                sr_img = np.repeat(sr_img, 3, axis=2)
                            elif hr_img.shape[2] == 1:
                                hr_img = np.repeat(hr_img, 3, axis=2)
                    
                    # Resize to same dimensions (resize SR to match HR size)
                    if sr_img.shape[:2] != hr_img.shape[:2]:
                        sr_img = cv2.resize(sr_img, (hr_img.shape[1], hr_img.shape[0]), interpolation=cv2.INTER_CUBIC)

                    # Calculate metrics
                    avg_psnr += Metrics.calculate_psnr(sr_img, hr_img)
                
                avg_psnr = avg_psnr / idx
                diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
                
                # Log validation results
                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(current_epoch, current_step, avg_psnr))
                tb_logger.add_scalar('psnr', avg_psnr, current_step)
                
            # Save checkpoint
            if current_step % opt['train']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/etdms.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-s', '--stage', type=int, choices=[1, 2, 0],
                        help='Train stage 1 (denoising), stage 2 (super-resolution) or both (0)', default=0)
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    
    # Parse configuration
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    
    # Logger setup
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])
    
    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None
    
    # Training stages
    if args.stage == 0 or args.stage == 1:
        train_stage1(opt, logger, tb_logger, wandb_logger)
    
    if args.stage == 0 or args.stage == 2:
        train_stage2(opt, logger, tb_logger, wandb_logger)