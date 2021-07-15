import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from torchvision.models import resnet34


class ResNet34(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet34(pretrained=False)
        self.model.fc = nn.Linear(in_features=512, out_features=1, bias=True)

        # loss
        self.criterion = nn.MSELoss()

    def forward(self, x):
        output = self.model(x).squeeze(1)
        return output

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = self.criterion(y, t)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = self.criterion(y, t)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
