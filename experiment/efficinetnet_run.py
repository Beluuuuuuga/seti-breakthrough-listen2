import os
import random
import numpy as np
import pandas as pd

# from PIL import Image
# from matplotlib import pyplot as plt

# # import seaborn as sns
# # import plotly.express as px
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn

# import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader

# from torch.optim.lr_scheduler import (
#     CosineAnnealingWarmRestarts,
#     CosineAnnealingLR,
#     ReduceLROnPlateau,
#     OneCycleLR,
# )
from torch.optim.lr_scheduler import CosineAnnealingLR

# from torch.optim.optimizer import Optimizer
# import torchvision.utils as vutils
import pytorch_lightning as pl

from pytorch_lightning import seed_everything
from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Config
SEED = 42
N_FOLDS = 5
TRAIN_FOLD = 0
TARGET_COL = "target"
# N_EPOCHS = 10
N_EPOCHS = 1
# BATCH_SIZE = 32
BATCH_SIZE = 16
# IMG_SIZE = 640
IMG_SIZE = 50
LR = 1e-4
MAX_LR = 5e-4
PRECISION = 16
MODEL = "efficientnet_b0"


def set_seed(seed=int):
    """Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY."""
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    seed_everything(seed)
    return random_state


# Dataset
class TrainDataset(Dataset):
    def __init__(self, df, test=False, transform=None):
        self.df = df
        self.test = test
        self.file_names = df["file_path"].values
        if not self.test:
            self.labels = df[TARGET_COL].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]

        image = np.load(file_path)[[0, 2, 4]]
        image = image.astype(np.float32)
        image = np.vstack(image).T
        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = image[np.newaxis, :, :]
            image = torch.from_numpy(image).float()
        if not self.test:
            label = torch.unsqueeze(torch.tensor(self.labels[idx]).float(), -1)
            return image, label
        else:
            return image


def get_transforms(*, data):

    if data == "train":
        return A.Compose(
            [
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                ToTensorV2(),
            ]
        )

    elif data == "valid":
        return A.Compose(
            [
                A.Resize(IMG_SIZE, IMG_SIZE),
                ToTensorV2(),
            ]
        )


# Data Module
class DataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, batch_size=8):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

    def setup(self, stage=None):
        self.train_dataset = TrainDataset(
            self.train_df, transform=get_transforms(data="train")
        )

        self.val_dataset = TrainDataset(
            self.val_df, transform=get_transforms(data="valid")
        )

        self.test_dataset = TrainDataset(
            self.test_df, transform=get_transforms(data="valid"), test=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )


def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Predictor(pl.LightningModule):
    def __init__(self, n_classes: int, n_training_steps=None, steps_per_epoch=None):
        super().__init__()
        self.n_classes = n_classes
        self.model = timm.create_model(MODEL, pretrained=True, in_chans=1)
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, n_classes)
        self.n_training_steps = n_training_steps
        self.steps_per_epoch = steps_per_epoch
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):

        x, y = batch

        # x, y_a, y_b, lam = mixup_data(x, y.view(-1, 1)) # mixup

        output = self(x)
        labels = y  # mixup
        # loss = mixup_criterion(self.criterion, output, y_a, y_b, lam) # mixup

        loss = self.criterion(output, y)

        try:
            auc = roc_auc_score(labels.detach().cpu(), output.sigmoid().detach().cpu())
            self.log("auc", auc, prog_bar=True, logger=True)
        except:
            pass
        return {"loss": loss, "predictions": output, "labels": labels}

    def training_epoch_end(self, outputs):

        preds = []
        labels = []

        for output in outputs:

            preds += output["predictions"]
            labels += output["labels"]

        labels = torch.stack(labels)
        preds = torch.stack(preds)

        train_auc = roc_auc_score(labels.detach().cpu(), preds.sigmoid().detach().cpu())
        self.log("mean_train_auc", train_auc, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        labels = y
        loss = self.criterion(output, y)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"predictions": output, "labels": labels}

    def validation_epoch_end(self, outputs):

        preds = []
        labels = []

        for output in outputs:

            preds += output["predictions"]
            labels += output["labels"]

        labels = torch.stack(labels)
        preds = torch.stack(preds)

        val_auc = roc_auc_score(labels.detach().cpu(), preds.sigmoid().detach().cpu())
        self.log("val_auc", val_auc, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x = batch
        output = self(x).sigmoid()
        return output

    def configure_optimizers(self):
        # optimizer = MADGRAD(self.parameters(), lr=LR)

        # scheduler = OneCycleLR(
        #     optimizer,
        #     epochs=N_EPOCHS,
        #     max_lr=MAX_LR,
        #     total_steps=self.n_training_steps,
        #     steps_per_epoch=self.steps_per_epoch,
        # )

        # 'T_max': 10,
        # 'min_lr': 1e-6,
        # optimizer = Ranger(self.parameters(), lr=params["lr"])
        optimizer = Adam(
            self.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )
        #  return torch.optim.Adam(self.parameters(), lr=0.02)

        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6, last_epoch=-1)

        return dict(optimizer=optimizer, lr_scheduler=scheduler)


def get_data():
    train = pd.read_csv("input/seti-breakthrough-listen/train_labels.csv")
    test = pd.read_csv("input/seti-breakthrough-listen/sample_submission.csv")

    def get_train_file_path(image_id):
        return "input/seti-breakthrough-listen/train/{}/{}.npy".format(
            image_id[0], image_id
        )

    def get_test_file_path(image_id):
        return "input/seti-breakthrough-listen/test/{}/{}.npy".format(
            image_id[0], image_id
        )

    # train["file_path"] = train["id"].apply(get_train_file_path)
    train["file_path"] = train["id"].apply(get_train_file_path)
    test["file_path"] = test["id"].apply(get_test_file_path)
    # return train, test
    return train[:100], test[:100]


def run_fold(train_fold, train, test):
    t_steps_per_epoch = (len(train) // N_EPOCHS) // BATCH_SIZE
    total_training_steps = t_steps_per_epoch * N_EPOCHS

    model = Predictor(
        steps_per_epoch=t_steps_per_epoch,
        n_training_steps=total_training_steps,
        n_classes=1,
    )

    data_module = DataModule(
        # train[train["fold"] != TRAIN_FOLD],  # train fold
        # train[train["fold"] == TRAIN_FOLD],  # val fold
        # train[train["fold"] == TRAIN_FOLD],  # test data, same as val for now
        train[train["fold"] != train_fold],  # train fold
        train[train["fold"] == train_fold],  # val fold
        train[train["fold"] == train_fold],  # test data, same as val for now
        batch_size=BATCH_SIZE,
    )

    early_stopping_callback = EarlyStopping(monitor="val_auc", mode="max", patience=2)
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{val_loss:.2f}-{other_metric:.2f}",
        # filename=fname,
        save_weights_only=True,
        save_top_k=None,
        monitor=None,
    )
    trainer = pl.Trainer(
        # checkpoint_callback=checkpoint_callback,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=N_EPOCHS,
        gpus=1,
        precision=PRECISION,
        # progress_bar_refresh_rate=5,
    )
    trainer.fit(model, data_module)

    device = torch.device("cuda")
    # Inference
    predictions = []
    test_predictions = []
    trained_model = Predictor.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, n_classes=1
    )
    trained_model.eval()
    trained_model.freeze()
    trained_model = trained_model.to(device)

    # valid pred
    valid = train[train["fold"] == train_fold]
    val_dataset = TrainDataset(valid, transform=get_transforms(data="valid"), test=True)
    dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    for item in tqdm(dataloader, position=0, leave=True):
        prediction = trained_model(item.to(device))
        predictions.append(prediction.flatten().sigmoid())
    valid_predictions = torch.cat(predictions).detach().cpu()
    final_valid_preds = valid_predictions.squeeze(-1).numpy()

    # test pred
    test_dataset = TrainDataset(test, transform=get_transforms(data="valid"), test=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=4
    )
    for item in tqdm(test_dataloader, position=0, leave=True):
        test_prediction = trained_model(item.to(device))
        test_predictions.append(test_prediction.flatten().sigmoid())
    test_predictions = torch.cat(test_predictions).detach().cpu()
    final_preds = test_predictions.squeeze(-1).numpy()
    return final_preds, final_valid_preds


def run(mode):
    random_state = set_seed(SEED)
    train, test = get_data()
    # print(train.shape)
    # print(train.shape)
    # exit()
    valid_idx_dic = {}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    train["fold"] = -1
    for fold_id, (_, val_idx) in enumerate(skf.split(train["id"], train["target"])):
        train.loc[val_idx, "fold"] = fold_id
        valid_idx_dic[fold_id] = val_idx

    oof = np.zeros((len(train),), dtype=np.float32)
    test_predictions = []
    for train_fold in range(5):  # 5
        test_pred, valid_pred = run_fold(train_fold, train, test)
        print(test_pred.shape, valid_pred.shape)
        test_predictions.append(test_pred)

        # valid idx
        idx_valid = valid_idx_dic[train_fold]
        oof[idx_valid] = valid_pred

    output_dir = "./"

    # OOF
    oof_df = pd.DataFrame()
    oof_df["id"] = train["id"]
    oof_df["target"] = oof
    oof_df.to_csv("oof.csv", index=False)

    # Submissions
    pred_mean = np.array(test_predictions).mean(axis=0)
    submission = pd.read_csv("input/seti-breakthrough-listen/sample_submission.csv")
    submission = submission[:100]
    submission["target"] = pred_mean
    submission.to_csv("sub.csv", index=False)


if __name__ == "__main__":
    mode = "train"
    run(mode)
