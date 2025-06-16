import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import modules
logger = logging.getLogger('base')

####################
# initialize
####################

def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))

####################
# define network
####################

# Generator
def define_G(opt):
    model_opt = opt['model']
    
    if model_opt['which_model_G'] == 'swinir':
        from .swinir_modules import denoiser
        model = denoiser.SwinIRDenoiser(
            img_size=model_opt['swinir']['img_size'],
            in_chans=model_opt['swinir']['in_chans'],
            window_size=model_opt['swinir']['window_size'],
            img_range=model_opt['swinir']['img_range'],
            depths=model_opt['swinir']['depths'],
            embed_dim=model_opt['swinir']['embed_dim'],
            num_heads=model_opt['swinir']['num_heads'],
            mlp_ratio=model_opt['swinir']['mlp_ratio'],
            upscale=model_opt['swinir']['upscale'],
            resi_connection=model_opt['swinir']['resi_connection'],
            pretrained=model_opt['swinir']['pretrained']
        )
        return model
        
    elif model_opt['which_model_G'] == 'clsr':
        from .clsr_modules import diffusion, enet
        
        model = enet.Net(
            in_channel=model_opt['enet']['in_channel'],
            out_channel=model_opt['enet']['out_channel'],
            norm_groups=model_opt['enet']['norm_groups'],
            inner_channel=model_opt['enet']['inner_channel'],
            channel_mults=model_opt['enet']['channel_multiplier'],
            attn_res=model_opt['enet']['attn_res'],
            res_blocks=model_opt['enet']['res_blocks'],
            dropout=model_opt['enet']['dropout'],
            image_size=model_opt['diffusion']['image_size']
        )
        
        netG = diffusion.GaussianDiffusion(
            model,
            image_size=model_opt['diffusion']['image_size'],
            channels=model_opt['diffusion']['channels'],
            loss_type='l1',    # L1 or L2
            conditional=model_opt['diffusion']['conditional'],
            schedule_opt=model_opt['beta_schedule']['train']
        )
        
        if opt['phase'] == 'train':
            init_weights(netG, init_type='orthogonal')
        if opt['gpu_ids'] and opt['distributed']:
            assert torch.cuda.is_available()
            netG = nn.DataParallel(netG)
        return netG
    
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model_opt['which_model_G']))
