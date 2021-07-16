import os
import sys

# sys.path = ['../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master',] + sys.path
import pandas as pd
import numpy as np

# from sklearn import metrics
# from tqdm import tqdm
import torch
import torch.nn as nn

# from efficientnet_pytorch import model as enet
# import random
# from sklearn.model_selection import StratifiedKFold

from torch.utils.data import DataLoader, Dataset


class ClassificationDataset:
    def __init__(self, image_paths, targets):
        self.image_paths = image_paths
        self.targets = targets

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = np.load(self.image_paths[item]).astype(float)

        targets = self.targets[item]

        return {
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),
        }


class CFG:
    apex = False
    debug = False
    print_freq = 500
    num_workers = 4
    model_name = "efficientnet-b4"
    size = 640
    scheduler = "CosineAnnealingLR"  # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    epochs = 20
    # factor=0.2 # ReduceLROnPlateau
    # patience=4 # ReduceLROnPlateau
    # eps=1e-6 # ReduceLROnPlateau
    T_max = 10  # CosineAnnealingLR
    # T_0=6 # CosineAnnealingWarmRestarts
    lr = 1e-3
    min_lr = 5e-5
    batch_size = 10
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 0
    target_size = 1
    target_col = "target"
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    train = True


class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df["file_path"].values
        self.labels = df[CFG.target_col].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        image = np.load(file_path)[[0, 2, 4]]  # shape: (3, 273, 256)
        image = image.astype(np.float32)
        image = np.vstack(image).transpose((1, 0))
        image = self.transform(image=image)["image"]
        label = torch.tensor(self.labels[idx]).float()
        return image, label


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df["file_path"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        image = np.load(file_path)[[0, 2, 4]]  # shape: (3, 273, 256)
        image = image.astype(np.float32)
        image = np.vstack(image).transpose((1, 0))
        image = self.transform(image=image)["image"]

        return image
