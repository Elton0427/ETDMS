import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def process_image(opt, image_path, output_path):
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((opt['datasets']['val']['l_resolution'], opt['datasets']['val']['l_resolution'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img_tensor = transform(img).unsqueeze(0).to(next(iter(opt['gpu_ids'])))
    
    # Stage 1: Denoising
    stage1_opt = opt.copy()
    stage1_opt['model'] = opt['stage1']
    denoiser = Model.create_model(stage1_opt)
    
    with torch.no_grad():
        denoised = denoiser(img_tensor)
    
    # Stage 2: Super-resolution
    stage2_opt = opt.copy()
    stage2_opt['model'] = opt['stage2']
    diffusion = Model.create_model(stage2_opt)
    diffusion.set_new_noise_schedule(opt['stage2']['beta_schedule']['val'], schedule_phase='val')
    
    # Prepare data
    data = {'LR': denoised}
    diffusion.feed_data(data)
    
    # Inference
    diffusion.test(continous=False)
    visuals = diffusion.get_current_visuals(need_LR=True)
    
    # Save results
    sr_img = Metrics.tensor2img(visuals['SR'][-1])
    lr_img = Metrics.tensor2img(visuals['LR'])
    
    os.makedirs(output_path, exist_ok=True)
    base_name = os.path.basename(image_path).split('.')[0]
    
    Metrics.save_img(lr_img, os.path.join(output_path, f"{base_name}_denoised.png"))
    Metrics.save_img(sr_img, os.path.join(output_path, f"{base_name}_sr.png"))
    
    return lr_img, sr_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/etdms.json',
                        help='JSON file for configuration')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('-o', '--output', type=str, default='results/etdms',
                        help='Output directory')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    
    # Parse configuration
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    
    # Setup GPU
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # Process input
    if os.path.isfile(args.input):
        # Single file
        process_image(opt, args.input, args.output)
    elif os.path.isdir(args.input):
        # All images in directory
        for file in os.listdir(args.input):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                input_path = os.path.join(args.input, file)
                process_image(opt, input_path, args.output)
    else:
        print(f"Error: {args.input} is not a valid file or directory")