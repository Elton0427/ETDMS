import logging
logger = logging.getLogger('base')

def create_model(opt):
    if opt['model']['which_model_G'] == 'swinir':
        from .swinir_model import SwinIRModel as M
    elif opt['model']['which_model_G'] == 'clsr':
        from .clsr_model import CLSRModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(opt['model']['which_model_G']))
    
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m