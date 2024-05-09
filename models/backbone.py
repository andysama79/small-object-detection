import torch
import torch.nn as nn

from swin_transformer import SwinTransformer

class Backbone(nn.Module):
    def __init__(self, config):
        super(Backbone, self).__init__()
        self.model = SwinTransformer(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_chans=config.in_channels,
            num_classes=config.num_classes,
            embed_dim=config.embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            qk_scale=config.qk_scale,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            ape=config.ape,
            patch_norm=config.patch_norm,
            use_checkpoint=config.use_checkpoint
        )

    def forward(self, x):
        return self.model(x)
