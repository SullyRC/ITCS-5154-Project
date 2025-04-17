# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 01:27:45 2025

@author: sulli
"""

import numpy as np
import pandas as pd
import seaborn as sns
from fastai.vision.all import *
import torch
import torch.nn as nn
import os
import zipfile

filename = './imagenet_files'

if not os.path.exists(filename):

    # Download the files
    os.system('kaggle datasets download dimensi0n/imagenet-256')

    # Extract it to this file
    with zipfile.ZipFile('C:imagenet-256.zip', 'r') as zip_ref:
        zip_ref.extractall(filename)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, depth=12, mlp_dim=3072):
        super().__init__()
        self.layers = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, batch_first=True)
              for _ in range(depth)]
        )

    def forward(self, x):
        return self.layers(x)

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, num_heads=12, depth=12, mlp_dim=3072):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.transformer = TransformerEncoder(embed_dim, num_heads, depth, mlp_dim)
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.mlp_head(x[:, 0])
    

def image_to_patches(img, patch_size=16):
    _, h, w = img.shape
    assert h % patch_size == 0 and w % patch_size == 0, "Image size must be divisible by patch size"
    patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.reshape(-1, patch_size * patch_size * 3)
    return patches

class PatchTransform(Transform):
    def __init__(self, patch_size=16):
        self.patch_size = patch_size

    def encodes(self, img: TensorImage):
        patches = image_to_patches(img, patch_size=self.patch_size)
        return patches
    
n_classes = len(os.listdir(filename))

# Define ImageNet normalization values
image_mean_stats = ( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Create DataBlock with PatchTransform
imagenet_dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    get_y = parent_label,
    splitter=RandomSplitter(valid_pct=0.2),
    item_tfms=[Resize(224), ToTensor(), PatchTransform(patch_size=16)],
    #batch_tfms=Normalize.from_stats(*image_mean_stats)
)

print("Loading data")

# Create DataLoader
dls = imagenet_dblock.dataloaders(filename, bs=32, num_workers=2)

print("Data loaded")

if __name__ == '__main__':
    model = VisionTransformer(num_classes=n_classes,depth=6).to(device)
    learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
    # Train the model
    learn.fit(5,  1e-5)