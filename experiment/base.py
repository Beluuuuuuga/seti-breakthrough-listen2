import os
import random
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.model_selection import KFold

# from vivid.metrics import regression_metrics
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

# !pip install -q pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import timm

from efficientnet_pytorch import EfficientNet

from hedgehog.model.resnet import ResNet34
from hedgehog.datamodule.atma_dataset import AtmaDataset


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    pl.seed_everything(seed)


PHOTO_DIR = "./dataset_atmaCup11/photos"
assert torch.cuda.is_available()

DEVICE = torch.device("cuda")


def to_img_path(object_id):
    # return os.path.join(photo_dir, f'{object_id}.jpg')
    return os.path.join(PHOTO_DIR, f"{object_id}.jpg")


def read_image(object_id):
    return Image.open(to_img_path(object_id))


def preprocessing():
    dataset_root = "."
    input_dir = os.path.join(dataset_root, "dataset_atmaCup11")
    photo_dir = os.path.join(input_dir, "photos")

    # output_dir = os.path.join(dataset_root, "outputs_tutorial#1")

    photo_pathes = glob(os.path.join(photo_dir, "*.jpg"))
    train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))
    return train_df, test_df


def get_train_valid(fold_num: int):
    train_df, test_df = preprocessing()
    train_meta_df = train_df[["target", "object_id"]].copy()
    train_meta_df["object_path"] = train_meta_df["object_id"].map(to_img_path)

    # split fold
    fold = KFold(n_splits=5, shuffle=True, random_state=510)
    cv = list(fold.split(X=train_df, y=train_df["target"]))
    idx_tr, idx_valid = cv[fold_num]
    train_df = train_meta_df.iloc[idx_tr]
    valid_df = train_meta_df.iloc[idx_valid]
    y_valid = train_meta_df["target"].values[idx_valid]
    return train_df, valid_df, test_df, idx_valid


class AtmaDataModule(pl.LightningDataModule):
    def __init__(self, train_df, valid_df, test_df):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

    def setup(self, stage):
        self.train_dataset = AtmaDataset(meta_df=self.train_df)
        self.valid_dataset = AtmaDataset(meta_df=self.valid_df, is_train=False)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=32, num_workers=4
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset, batch_size=32, num_workers=4
        )


def create_metadata(input_df):
    out_df = input_df[["object_id"]].copy()
    out_df["object_path"] = input_df["object_id"].map(to_img_path)

    if "target" in input_df:
        out_df["target"] = input_df["target"]

    return out_df


def predict(model, loader) -> np.ndarray:
    model.to(DEVICE)

    model.eval()

    predicts = []

    for x_i, y_i in loader:
        with torch.no_grad():
            output = model(x_i.to(DEVICE))

        predicts.extend(output.data.cpu().numpy())

    pred = np.array(predicts).reshape(-1)
    return pred


def train(fold_num: int):
    model = ResNet34()

    train_df, valid_df, test_df, idx_valid = get_train_valid(fold_num)
    data_module = AtmaDataModule(train_df, valid_df, test_df)

    dir_name = "output"
    fname = "resnet_fold_" + str(fold_num)
    model_path = dir_name + "/" + fname + ".ckpt"
    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_name,
        # filename="{epoch}-{val_loss:.2f}-{other_metric:.2f}",
        filename=fname,
        save_weights_only=True,
        save_top_k=None,
        monitor=None,
    )
    #     logger = CSVLogger(save_dir="logs", name="my_exp")
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=50,
        callbacks=[
            checkpoint_callback,
            EarlyStopping("val_loss", patience=10, verbose=True),
        ],
    )
    trainer.fit(model, data_module)

    # 検証
    model.load_state_dict(torch.load(model_path)["state_dict"], strict=False)
    valid_dataset = AtmaDataset(meta_df=valid_df, is_train=False)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=32, drop_last=False, num_workers=4
    )
    valid_pred = predict(model, valid_loader)

    return (valid_pred, idx_valid)


def inference(fold_num: int):
    model = ResNet34()
    train_df, valid_df, test_df, idx_valid = get_train_valid(fold_num)
    # 推論
    test_meta_df = create_metadata(test_df)
    # 学習時のデータ拡張はオフにしたいので is_train=False としている
    test_dataset = AtmaDataset(meta_df=test_meta_df, is_train=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=32, drop_last=False, num_workers=4
    )

    dir_name = "output"
    fname = "resnet_fold_" + str(fold_num)
    model_path = dir_name + "/" + fname + ".ckpt"
    model.load_state_dict(torch.load(model_path)["state_dict"], strict=False)
    test_pred = predict(model, test_loader)
    return test_pred


def calculate_metrics(y_true, y_pred) -> dict:
    """正解ラベルと予測ラベルから指標を計算する"""
    # return regression_metrics(y_true, y_pred)

    return {"rmse": mean_squared_error(y_true, y_pred) ** 0.5}


def main(mode):
    seed_torch(seed=42)
    if mode == "train":
        train_df, _ = preprocessing()
        oof = np.zeros((len(train_df),), dtype=np.float32)

        # fold_num = 0
        for fold_num in range(5):
            print(fold_num)
            valid_pred, idx_valid = train(fold_num)
            oof[idx_valid] = valid_pred

        print(calculate_metrics(train_df["target"], oof))
        oof_df = pd.DataFrame()
        oof_df["target_origin"] = train_df["target"]
        oof_df["target_pred"] = oof
        oof_df.to_csv("resnet_base_1.csv", index=False)
    else:
        test_predictions = []
        for fold_num in range(5):
            print(fold_num)
            test_pred = inference(fold_num)
            test_predictions.append(test_pred)
        # すべての予測の平均値を使う
        output_dir = "./"
        pred_mean = np.array(test_predictions).mean(axis=0)
        pd.DataFrame({"target": pred_mean}).to_csv(
            os.path.join(output_dir, "0003__submission.csv"), index=False
        )


if __name__ == "__main__":
    mode = "test"
    main(mode)
