import logging
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import model.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')

class CLSRModel(BaseModel):
    def __init__(self, opt):
        super(CLSRModel, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None
        
        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        
        if self.opt['phase'] == 'train':
            self.netG.train()
            # CLSR specific optimizer settings
            optim_params = list(self.netG.parameters())
            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        # CLSR forward pass: HQLR -> HQHR
        l_pix = self.netG(self.data)  # Diffusion loss
        l_pix.backward()
        self.optG.step()
        
        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['SR'], continous)
            else:
                self.SR = self.netG.super_resolution(self.data['SR'], continous)
        self.netG.train()

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

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
            self.opt['path']['checkpoint'], 'I{}_E{}_clsr.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_clsr_opt.pth'.format(iter_step, epoch))
        # Save network
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        torch.save(state_dict, gen_path)
        # Save optimizer
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': self.optG.state_dict()}
        torch.save(opt_state, opt_path)
        logger.info('Saved CLSR model in [{:s}] ...'.format(gen_path))

    def set_loss(self):
        """Initialize the loss function for the diffusion model"""
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)
    
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