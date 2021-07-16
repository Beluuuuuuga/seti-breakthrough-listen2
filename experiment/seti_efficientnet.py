import sys

# sys.path.append('../../input/pytorch-image-models/')
# sys.path.append('../../input/EfficientNet-PyTorch/')

import os
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from tqdm.auto import tqdm
from functools import partial

import cv2
from PIL import Image
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

# import timm
from efficientnet_pytorch import EfficientNet

from torch.cuda.amp import autocast, GradScaler

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from seti_dataset import TrainDataset

# from sam import SAM  # optim

train = pd.read_csv("input/seti-breakthrough-listen/train_labels.csv")
test = pd.read_csv("input/seti-breakthrough-listen/sample_submission.csv")


def get_train_file_path(image_id):
    return "input/seti-breakthrough-listen/train/{}/{}.npy".format(
        image_id[0], image_id
    )


def get_test_file_path(image_id):
    return "input/seti-breakthrough-listen/test/{}/{}.npy".format(image_id[0], image_id)


train["file_path"] = train["id"].apply(get_train_file_path)
test["file_path"] = test["id"].apply(get_test_file_path)


parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=512)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batchs", type=int, default=32)
parser.add_argument("--fold_n", type=int, default=5)
parser.add_argument("--model", type=str, default="efficientnet-b0")
parser.add_argument("--output", type=str, default="v1")
args = parser.parse_args()


OUTPUT_DIR = "./outputs/" + args.output + "/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class CFG:
    apex = False
    debug = False
    print_freq = 500
    num_workers = 4
    # model_name = "efficientnet-b0"
    model_name = args.model
    # size = 512
    size = args.size
    scheduler = "CosineAnnealingLR"  # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    epochs = args.epochs
    # epochs=3 # change
    # factor=0.2 # ReduceLROnPlateau
    # patience=4 # ReduceLROnPlateau
    # eps=1e-6 # ReduceLROnPlateau
    T_max = 10  # CosineAnnealingLR
    # T_0=6 # CosineAnnealingWarmRestarts
    lr = 1e-3
    # lr=1e-5 # change
    min_lr = 5e-5
    batch_size = args.batchs
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 0
    target_size = 1
    target_col = "target"
    n_fold = args.fold_n
    trn_fold = [i for i in range(args.fold_n)]
    train = True


if CFG.debug:
    CFG.epochs = 1
    train = train.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)


def get_score(y_true, y_pred):
    score = roc_auc_score(y_true, y_pred)
    return score


def init_logger(log_file=OUTPUT_DIR + "train.log"):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(seed=CFG.seed)

Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(Fold.split(train, train[CFG.target_col])):
    train.loc[val_index, "fold"] = int(n)
train["fold"] = train["fold"].astype(int)
print(train.groupby(["fold", "target"]).size())


def get_transforms(*, data):

    if data == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size),
                A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.CoarseDropout(p=0.5),
                # A.Cutout(p=0.5),
                ToTensorV2(),
            ]
        )

    elif data == "valid":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size),
                ToTensorV2(),
            ]
        )


class CustomModel(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.model = EfficientNet.from_pretrained(self.cfg.model_name)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=3, bias=False)
        self.n_features = self.model._fc.in_features
        self.classifier = nn.Linear(self.n_features, self.cfg.target_size)
        self.model._fc = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.model(x)
        output = self.classifier(x)
        return output


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    if CFG.apex:
        scaler = GradScaler()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images, labels_a, labels_b, lam = mixup_data(
            images, labels.view(-1, 1), alpha=1.0, use_cuda=True
        )  # alpha can change 0.0~1.0

        images = images.to(device, dtype=torch.float)
        labels_a = labels_a.to(device, dtype=torch.float)
        labels_b = labels_b.to(device, dtype=torch.float)

        batch_size = labels.size(0)
        if CFG.apex:
            with autocast():
                y_preds = model(images)
                loss = mixup_criterion(criterion, y_preds, labels_a, labels_b, lam)
        else:
            y_preds = model(images)
            loss = mixup_criterion(criterion, y_preds, labels_a, labels_b, lam)
        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        if CFG.apex:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm
        )
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            if CFG.apex:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "Grad: {grad_norm:.4f}  "
                "LR: {lr:.6f}  ".format(
                    epoch + 1,
                    step,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    grad_norm=grad_norm,
                    lr=scheduler.get_lr()[0],
                )
            )
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds.view(-1), labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.sigmoid().to("cpu").numpy())
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                "EVAL: [{0}/{1}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    step,
                    len(valid_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    remain=timeSince(start, float(step + 1) / len(valid_loader)),
                )
            )
    predictions = np.concatenate(preds)
    return losses.avg, predictions


def train_loop(folds, fold):

    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_col].values

    train_dataset = TrainDataset(train_folds, transform=get_transforms(data="train"))
    valid_dataset = TrainDataset(valid_folds, transform=get_transforms(data="valid"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(optimizer):
        if CFG.scheduler == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=CFG.factor,
                patience=CFG.patience,
                verbose=True,
                eps=CFG.eps,
            )
        elif CFG.scheduler == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1
            )
        elif CFG.scheduler == "CosineAnnealingWarmRestarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1
            )
        else:
            scheduler = None
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    # model = CustomModel(CFG, fold,pretrained=True)
    model = CustomModel(CFG, pretrained=True)
    model.to(device)

    optimizer = Adam(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False
    )
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss()

    best_score = 0.0
    best_loss = np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(
            train_loader, model, criterion, optimizer, epoch, scheduler, device
        )

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            if (epoch + 1) <= CFG.T_max:
                scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

        # scoring
        score = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        LOGGER.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        LOGGER.info(f"Epoch {epoch+1} - Score: {score:.4f}")

        if score > best_score:
            best_score = score
            LOGGER.info(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "preds": preds},
                OUTPUT_DIR + f"{CFG.model_name}_fold{fold}_best_score.pth",
            )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f"Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "preds": preds},
                OUTPUT_DIR + f"{CFG.model_name}_fold{fold}_best_loss.pth",
            )

    valid_folds["preds"] = torch.load(
        OUTPUT_DIR + f"{CFG.model_name}_fold{fold}_best_loss.pth",
        map_location=torch.device("cpu"),
    )["preds"]

    return valid_folds


def make_oof(score_loss="loss"):
    # score_loss: 'loss' or 'score'
    def get_result(result_df):
        preds = result_df["preds"].values
        labels = result_df[CFG.target_col].values
        score = get_score(labels, preds)
        print(f"Score: {score:<.4f}")

    oof_df = pd.DataFrame()
    for fold in range(CFG.n_fold):
        val_idx = train[train["fold"] == fold].index
        valid_folds = train.loc[val_idx].reset_index(drop=True)
        if score_loss == "loss":
            valid_folds["preds"] = torch.load(
                OUTPUT_DIR + f"{CFG.model_name}_fold{fold}_best_loss.pth",
                map_location=torch.device("cpu"),
            )["preds"]
        elif score_loss == "score":
            valid_folds["preds"] = torch.load(
                OUTPUT_DIR + f"{CFG.model_name}_fold{fold}_best_score.pth",
                map_location=torch.device("cpu"),
            )["preds"]
        _oof_df = valid_folds
        oof_df = pd.concat([oof_df, _oof_df])
        get_result(_oof_df)
        oof_df.to_csv(OUTPUT_DIR + "oof_df.csv", index=False)


def main():

    """
    Prepare: 1.train
    """

    def get_result(result_df):
        preds = result_df["preds"].values
        labels = result_df[CFG.target_col].values
        score = get_score(labels, preds)
        LOGGER.info(f"Score: {score:<.4f}")

    if CFG.train:
        # train
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            # if fold > 0: continue
            if fold in CFG.trn_fold:
                _oof_df = train_loop(train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        # CV result
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv(OUTPUT_DIR + "oof_df.csv", index=False)


if __name__ == "__main__":
    main()
