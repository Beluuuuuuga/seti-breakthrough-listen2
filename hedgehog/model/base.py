import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import pytorch_lightning as pl


class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * 56 * 56, 1)

    def forward(self, x):
        h = self.conv1(x)  # (bs, 3, 224, 224) -> (bs, 32, 224, 224)
        h = self.bn1(h)  # (bs, 32, 224, 224) -> (bs, 32, 224, 224)
        h = F.relu(h)  # (bs, 32, 224, 224) -> (bs, 32, 224, 224)
        h = F.max_pool2d(
            h, kernel_size=2, stride=2
        )  # (bs, 32, 224, 224) -> (bs, 32, 112, 112)

        h = self.conv2(h)  # (bs, 32, 112, 112) -> (bs, 64, 112, 112)
        h = self.bn2(h)  # (bs, 64, 112, 112) -> (bs, 64, 112, 112)
        h = F.relu(h)  # (bs, 64, 112, 112) -> (bs, 64, 112, 112)
        h = F.max_pool2d(
            h, kernel_size=2, stride=2
        )  # (bs, 64, 112, 112) -> (bs, 64, 56, 56)

        h = h.view(-1, 64 * 56 * 56)  #  (bs, 64, 56, 56) -> (bs, 64*56*56)
        h = self.fc(h)
        return h

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = torch.sqrt(F.mse_loss(y, t))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = torch.sqrt(F.mse_loss(y, t))
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer
