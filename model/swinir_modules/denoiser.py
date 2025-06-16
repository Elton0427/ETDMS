import torch
import torch.nn as nn
from .network_swinir import SwinIR 

class SwinIRDenoiser(nn.Module):
    def __init__(self, img_size=64, in_chans=3, window_size=8, img_range=1.0, depths=[6, 6, 6, 6, 6, 6],
                 embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upscale=1, resi_connection='1conv',
                 pretrained=None, pretrained_noise_level=15):
        super(SwinIRDenoiser, self).__init__()
        
        self.model = SwinIR(
            img_size=img_size,
            patch_size=1,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            upscale=upscale,
            img_range=img_range,
            upsampler='',
            resi_connection=resi_connection
        )
        
        if pretrained is not None:
            self.model.load_state_dict(torch.load(pretrained), strict=True)
            print(f"Loaded pretrained SwinIR model from {pretrained}")
    
    def forward(self, x):
        return self.model(x)