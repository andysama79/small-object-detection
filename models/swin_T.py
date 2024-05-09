# %%
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as f

import einops
from einops import rearrange, repeat

class FeedForward(nn.Module):
    def __init__(self, hidden_dim, dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(), # activation function - gaussian error linear unit
            nn.Linear(hidden_dim, dim)
        )
        

    def forward(self, x):
        return self.network(x)

class CyclicShift(nn.Module):
    def __init__(self, disp):
        super().__init__()
        self.disp = disp

    def forward(self, x):
        return torch.roll(x, shifts=(self.disp, self.disp), dims=(1, 2))

def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask

def get_rel_dist(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))

    dist = indices[None, :, :] - indices[:, None, :]
    return dist

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, shifted, window_size, rel_pos_emb):
        super().__init__()
        print("WindowAttention")
        inner_dim = head_dim * num_heads    #will be equal to number of channels sucj as (32*3 = 96 for the first layer)
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5   #scale factor for the attention mechanism (1/sqrt(d_k))
        self.window_size = window_size  #for now, we will keep it as 7
        self.shifted = shifted
        self.rel_pos_emb = rel_pos_emb
        if self.shifted:
            disp = window_size // 2
            self.cyclic_shift = CyclicShift(-disp)
            self.cyclic_shift_rev = CyclicShift(disp)
        
            self.top_bottom_mask = nn.Parameter(create_mask(window_size=window_size, displacement=disp, upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=disp, upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        # self.pos_emb = nn.Parameter(torch.randn(window_size**2, window_size**2))
        # this is realtive position embedding for the window size
        if self.rel_pos_emb:
            self.rel_ind = get_rel_dist(window_size) + window_size - 1
            self.pos_emb = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_emb = nn.Parameter(torch.randn(window_size**2, window_size**2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, **kwargs):
        if self.shifted:
            x = self.cyclic_shift(x)

        batch, n_height, n_width, _, h = *x.shape, self.num_heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        num_window_h = n_height//self.window_size
        num_window_w = n_width//self.window_size

        q,k,v = map(lambda t: rearrange(t, 'batch (num_window_h w_h) (num_window_w w_w) (h d) -> batch h (num_window_h num_window_w) (w_h w_w) d', h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        # this is a dot product similarity approch
        # dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        # A better approch would be use the cosine similarity pointed out in the version 2 of the swin_t paper
        #  better because : it an be beneficial for object detection tasks when there are many similar objects (e.g., different bird species) that need to be distinguished.
        
        self.tau = nn.Parameter(torch.tensor(0.02), requires_grad=True)
        q = f.normalize(q, p=2, dim=-1)
        k = f.normalize(k, p=2, dim=-1)
        
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) / self.tau
        
        
        # here the possition embedding is added to all the rows
        # dots += self.pos_emb
        if self.rel_pos_emb:
            temp1 = self.rel_ind[:,:,0]
            dots += self.pos_emb[self.rel_ind[:,:,0], self.rel_ind[:,:,1]]
        else:
            dots += self.pos_emb

        if self.shifted:
            #here mask are being added to the last rows be it bpttom or right
            dots[:,:,-num_window_w:]+=self.top_bottom_mask
            dots[:,:,num_window_w-1::num_window_w] += self.left_right_mask

        attntion = dots.softmax(dim=-1)
        out = einsum('b h w i j, b h w j d -> b h w i d', attntion, v)
        out = rearrange(out, 'b h (num_window_h num_window_w) (w_h w_w) d -> b (num_window_h w_h) (num_window_w w_w) (h d)', h=h, w_h=self.window_size, w_w=self.window_size, num_window_h=num_window_h, num_window_w=num_window_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_shift_rev(out)

        return out

class PreNorm(nn.Module):
    def __init__(self, fn, dim):
        super().__init__()
        print("attention block PreNorm")
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))  #this is different from the paper1 of swin whre the preNorm is applied before the mlp and attention

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        print("atten_block Residual")
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class Swin_Block(nn.Module):
    def __init__(self, dim, num_heads, head_dim, mlp_dim, shifted,window_size, rel_pos_emb):
        super().__init__()
        print("Swin_Block")
        self.attention_block = Residual(PreNorm(WindowAttention(dim=dim, num_heads=num_heads, head_dim=head_dim, shifted=shifted, window_size=window_size, rel_pos_emb=rel_pos_emb), dim))
        self.mlp_block = Residual(PreNorm(FeedForward(hidden_dim=mlp_dim, dim=dim), dim))
    

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x
        
class PatchMerging_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, down_scaling_factor):
        super().__init__()
        self.patch_merge = nn.Conv2d(in_channels, out_channels, kernel_size=down_scaling_factor, stride=down_scaling_factor, padding=0)

    def forward(self, x):
        x = self.patch_merge(x).permute(0, 2, 3, 1)
        return x

class StageModule(nn.Module):
    def __init__(self, in_channel, hid_dim, layers, down_scaling_factor, num_heads, head_dim, window_size, rel_pos_emb):
        super().__init__()
        assert layers % 2 == 0, 'number of layers should be even'
        self.patch_partition = PatchMerging_Conv(in_channels=in_channel, out_channels=hid_dim, down_scaling_factor=down_scaling_factor)
        self.layers = nn.ModuleList([])
        for _ in range(layers//2):
            self.layers.append(nn.ModuleList([
                Swin_Block(dim=hid_dim, num_heads=num_heads, head_dim=head_dim, mlp_dim = hid_dim*4, shifted=False ,window_size=window_size, rel_pos_emb=rel_pos_emb),
                Swin_Block(dim=hid_dim, num_heads=num_heads, head_dim=head_dim, mlp_dim = hid_dim*4, shifted=True ,window_size=window_size, rel_pos_emb=rel_pos_emb),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular, shifted in self.layers:
            x = regular(x)
            x = shifted(x)
        print("returning from stage module")
        return x.permute(0,3,1,2)

class SwinTransformer(nn.Module):
    def __init__(self, *, hid_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=7, down_scaling_fact=(4,2,2,2), rel_pos_emb = True):
      super().__init__()
      self.stage1 = StageModule(in_channel = channels, hid_dim=hid_dim, layers=layers[0], down_scaling_factor=down_scaling_fact[0], num_heads=heads[0], head_dim=head_dim, window_size=window_size, rel_pos_emb=rel_pos_emb)
      self.stage2 = StageModule(in_channel = hid_dim, hid_dim=hid_dim*2, layers=layers[1], down_scaling_factor=down_scaling_fact[1], num_heads=heads[1], head_dim=head_dim, window_size=window_size, rel_pos_emb=rel_pos_emb)
      self.stage3 = StageModule(in_channel = hid_dim*2, hid_dim=hid_dim*4, layers=layers[2], down_scaling_factor=down_scaling_fact[2], num_heads=heads[2], head_dim=head_dim, window_size=window_size, rel_pos_emb=rel_pos_emb)
      self.stage4 = StageModule(in_channel = hid_dim*4, hid_dim=hid_dim*8, layers=layers[3], down_scaling_factor=down_scaling_fact[3], num_heads=heads[3], head_dim=head_dim, window_size=window_size, rel_pos_emb=rel_pos_emb)

    #   self.mlp_head = nn.Sequential(
    #         nn.LayerNorm(hid_dim*8),
    #         nn.Linear(hid_dim*8, num_classes)
    #     )
    
    def forward(self, image):
        x = self.stage1(image)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        # x = x.mean(dim=[2,3])
        # return self.mlp_head(x)
        return x

def swin_t(hid_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hid_dim=hid_dim, layers=layers, heads=heads, **kwargs)