import logging
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import model.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')

class SwinIRModel(BaseModel):
    def __init__(self, opt):
        super(SwinIRModel, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        
        # set loss and load resume state
        self.set_loss()
        
        if self.opt['phase'] == 'train':
            self.netG.train()
            # SwinIR specific optimizer settings
            optim_params = list(self.netG.parameters())
            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        
        self.load_network()
        self.print_network()

    def load_network(self):
        load_path = self.opt.get('path', {}).get('pretrained_model_G', None)
        if load_path is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path))
            self.load_network_from_path(self.netG, load_path, 
                                       self.opt.get('path', {}).get('strict_load', True))

    def load_network_from_path(self, network, load_path, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def set_loss(self):
        if self.opt['train'] and self.opt['train']['pixel_criterion']:
            loss_type = self.opt['train']['pixel_criterion']
            if loss_type == 'l1':
                self.loss_func = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.loss_func = nn.MSELoss().to(self.device)
            elif loss_type == 'smooth_l1':
                self.loss_func = nn.SmoothL1Loss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] not recognized.'.format(loss_type))
        else:
            self.loss_func = nn.L1Loss().to(self.device)  # default to L1 loss
        
        logger.info('Using loss function: {}'.format(self.loss_func.__class__.__name__))

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        # SwinIR forward pass: LQLR -> HQLR
        pred_hqlr = self.netG(self.data['LR'])  # Low quality LR -> High quality LR
        l_pix = self.loss_func(pred_hqlr, self.data['HR'])  # HR here is actually HQLR
        l_pix.backward()
        self.optG.step()
        
        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            self.SR = self.netG(self.data['LR'])
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if need_LR:
            out_dict['LR'] = self.data['LR'].detach()[0].float().cpu()
        out_dict['SR'] = self.SR.detach()[0].float().cpu()
        if 'HR' in self.data:
            out_dict['HR'] = self.data['HR'].detach()[0].float().cpu()
        return out_dict

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_swinir.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_swinir_opt.pth'.format(iter_step, epoch))
        # Save network
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        torch.save(state_dict, gen_path)
        # Save optimizer
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': self.optG.state_dict()}
        torch.save(opt_state, opt_path)
        logger.info('Saved SwinIR model in [{:s}] ...'.format(gen_path))


    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        logger.info('SwinIR does not use noise scheduling, ignoring set_new_noise_schedule call')
        pass