import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch import einsum

import einops
from einops import rearrange, repeat

from einops.layers.torch import Rearrange

from neck import *

class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_scaling=2):
        super(FPNBlock, self).__init__()
        self.up_scaling = up_scaling
        self.lateral_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.topdown_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.upmerge = UpMerging(in_channels=in_channels, up_scaling_factor=up_scaling)
    
    def forward(self, x):
        # x = F.interpolate(x, scale_factor=self.up_scaling, mode='nearest')
        # print("X shape:", x.shape, "Features shape:", features.shape)
        # x = torch.cat([x, features], dim=1)
        # x = x + features
        x = x.permute(0, 3, 1, 2)
        # x = self.upmerge(x.permute(0, 2, 3, 1))
        x = self.lateral_conv(x)
        # print("X shape: ", x.shape)
        x = x + self.topdown_conv(x)
        # print("X shape: ", x.shape)con
        x = self.upmerge(x)
        x = torch.cat([x, x], dim=3)
        # x = self.upmerge(x)
        # x = self.norm(x)
        # x = self.act(x)
        return x#.permute(0, 2, 3, 1)

class SwinFPNNeck(nn.Module):
    def __init__(self, *, hid_dim, layers, heads, channels, num_classes=1, head_dim=32, window_size=2, up_scaling_fact=(4, 2, 2, 2), rel_pos_emb=True):
        super(SwinFPNNeck, self).__init__()
        
        self.stage1 = StageModule(in_channel = channels, hid_dim=hid_dim, layers=layers[0], up_scaling_factor=up_scaling_fact[0], num_heads=heads[0], head_dim=head_dim, window_size=window_size, rel_pos_emb=rel_pos_emb)
        self.stage2 = StageModule(in_channel = channels//(2), hid_dim=hid_dim*2, layers=layers[1], up_scaling_factor=up_scaling_fact[1], num_heads=heads[1], head_dim=head_dim, window_size=window_size, rel_pos_emb=rel_pos_emb)
        self.stage3 = StageModule(in_channel = channels//(2*2), hid_dim=hid_dim*4, layers=layers[2], up_scaling_factor=up_scaling_fact[2], num_heads=heads[2], head_dim=head_dim, window_size=window_size, rel_pos_emb=rel_pos_emb)
        self.stage4 = StageModule(in_channel = channels//(2*2*2), hid_dim=hid_dim*8, layers=layers[3], up_scaling_factor=up_scaling_fact[3], num_heads=heads[3], head_dim=head_dim, window_size=window_size, rel_pos_emb=rel_pos_emb, upsample=False)
        
        self.fpn1 = FPNBlock(channels, channels//2, up_scaling=up_scaling_fact[0])
        self.fpn2 = FPNBlock(channels//2, channels//4, up_scaling=up_scaling_fact[1])
        self.fpn3 = FPNBlock(channels//4, channels//8, up_scaling=up_scaling_fact[2])

    #   self.mlp_head = nn.Sequential(
    #         nn.LayerNorm(hid_dim*8),
    #         nn.Linear(hid_dim*8, num_classes)
    #     )
    
    def forward(self, x, feature_maps):
        x1 = self.stage1(x + feature_maps[3]) # 384
        x2 = self.fpn1(x + feature_maps[3]) # 384
        # print("X1 shape:", x1.shape, "X2 shape:", x2.shape)
        x = x1 + x2
        
        x1 = self.stage2(x + feature_maps[2]) # 192
        x2 = self.fpn2(x + feature_maps[2])
        x = x1 + x2
        
        x1 = self.stage3(x + feature_maps[1]) # 96
        x2 = self.fpn3(x + feature_maps[1])
        x = x1 + x2
        
        x = self.stage4(x + feature_maps[0]) # 96
        
        return x
        