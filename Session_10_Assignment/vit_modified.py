# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

!pip install einops
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(dim, hidden_dim),
            nn.Conv2d(dim, hidden_dim,1),
            nn.GELU(),
            nn.Dropout(dropout),
            # nn.Linear(hidden_dim, dim),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = rearrange(x, 'b p c -> c p b')
        x = self.net(x)
        x = rearrange(x, 'c p b -> b p c')
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_qkv = nn.Sequential(
            Rearrange('b p dim -> dim b p'),
            nn.Conv2d(dim,inner_dim * 3, 1),
            Rearrange('i_dim b p -> b p i_dim')

        )
        self.to_out = nn.Sequential(
            Rearrange('b p i_dim -> i_dim b p'),
            nn.Conv2d(inner_dim, dim, 1),
            Rearrange('dim b p -> b p dim'),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        ## input x = (b , patch + 1 , dim = 128) -> output qkv = (b, patch + 1 , dim_heads * heads * 3)
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) ### separating into different heads

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')### combining all the heads
        # print(out.shape)

        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            # input(batch,channel,heightImage, widthImage) -> output(batch, h*w of patch, dim_embedding_of_patch)
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),       ###. h* p1 = height of original img therefor h = height of the patch , output = batch, dim of patches, patch dim(type of embedding for each patch).
            # nn.Linear(patch_dim, dim),
            nn.Conv2d(3, dim , kernel_size = (patch_height, patch_width), stride = (patch_height, patch_width)), # output(b, num of patches , patch_dim_embedding)
            Rearrange('b c h w -> b (h w) c')
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('(b c) dim -> dim b c', c = 1),
            nn.Conv2d(dim, num_classes, 1),
            Rearrange('class b c -> (b c) class')
        )

    def forward(self, img):
        # print(12)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        # print(b,n) ## b = batch number, n = number of patches = 63 in this case.
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        cls = x[ : , 0] ## extracts the cls token from the x

        cls = self.to_latent(cls)
        cls = self.mlp_head(cls)
        return cls