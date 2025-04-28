# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 01:27:45 2025

@author: sulli
"""

import numpy as np
import pandas as pd
import seaborn as sns
from fastai.vision.all import *
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.progress import CSVLogger
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


# Patch embed class
class PatchEmbedding(nn.Module):
    # Projects an patches into a lower dimension space using a Conv2d layer
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

# Transformer encoder class for a sequential tranformer model
class TransformerEncoder(nn.Module):
    # Sequential Transformer Encoder Layers of depth d
    def __init__(self, embed_dim=768, num_heads=12, depth=12, mlp_dim=1000):
        super().__init__()
        self.layers = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, batch_first=True)
              for _ in range(depth)]
        )

    def forward(self, x):
        return self.layers(x)


# Our model class
class VisionTransformer(nn.Module):
    # Embeds the image into patches then parses through the tranformer
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, num_heads=12, depth=12, mlp_dim=1000):
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
    

# Preprocessing that turns an image into a patch
def image_to_patches(img, patch_size=16):
    _, h, w = img.shapez
    assert h % patch_size == 0 and w % patch_size == 0, "Image size must be divisible by patch size"
    patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.reshape(-1, patch_size * patch_size * 3)
    return patches

# Batch transform images
class PatchTransform(Transform):
    def __init__(self, patch_size=16):
        self.patch_size = patch_size

    @torch.no_grad()
    def encodes(self, img: TensorImage):
        patches = image_to_patches(img, patch_size=self.patch_size)
        return patches
    
# Classes in our dataset
n_classes = len(os.listdir(filename))

# Build the dataloader if we don't have it on disk
if not os.path.exists('image_data_loader.pkl'):
    # Create DataBlock with PatchTransform
    imagenet_dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y = parent_label,
        splitter=RandomSplitter(valid_pct=0.2),
        item_tfms=[Resize(224), 
                   ToTensor(), PatchTransform(patch_size=16)],
        batch_tfms=[*aug_transforms()]
        )

    print("Loading data")

    # Create DataLoader
    dls = imagenet_dblock.dataloaders(filename, bs=100,
                                      num_workers=0, persistent_workers=False)
    print("Data loaded")

    with open('image_data_loader.pkl', 'wb') as handle:
        pickle.dump(dls, handle)

# Otherwise load it
else:
    with open('image_data_loader.pkl', 'rb') as handle:
        dls = pickle.load(handle)


# Train our model
if __name__ == '__main__':
    model = VisionTransformer(num_classes=n_classes,depth=8, mlp_dim=1000).to(device)
    learn = Learner(dls, model,
                    loss_func=CrossEntropyLossFlat(),
                    metrics=accuracy)
    
    # Create learning rate scheduler
    sched = {'lr': SchedExp(1e-4,1e-5)}
    
    # Train the model
    learn.fit(20,  cbs=[ParamScheduler(sched),
                       #SaveModelCallback(every_epoch=True,
                       #                        fname='model_checkpoint'),
                              CSVLogger(fname='Training_Logs.csv')]
              )
    
    learn.export('learner.pkl')
