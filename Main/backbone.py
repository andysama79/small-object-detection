import torch
import torch.nn as nn

from swin_transformer import SwinTransformer

class Backbone(nn.Module):
    def __init__(self, hid_dim, layers, heads, **kwargs):
        super(Backbone, self).__init__()
        self.model = SwinTransformer(hid_dim=hid_dim, layers=layers, heads=heads, **kwargs)

    def forward(self, x):
        return self.model(x)
