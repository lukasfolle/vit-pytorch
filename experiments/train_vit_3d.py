import torch
from torch.nn import functional as F
import pytorch_lightning as pl

from vit_pytorch import ViT
from experiments.Dataset import CovidCT


class ViTModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.vit = ViT(image_size=256, image_depth=24, patch_size=32, patch_depth=3, num_classes=3, dim=1024, depth=6,
                       heads=8, mlp_dim=2048, dropout=0.1, emb_dropout=0.1)

    def training_step(self, batch, batch_idx):
        x, y = batch["volume"], batch["label"]
        y_hat = self.vit(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["volume"], batch["label"]
        y_hat = self.vit(x)
        loss = F.cross_entropy(y_hat, y)

        self.log('val/loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch["volume"], batch["label"]
        y_hat = self.vit(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test/loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


if __name__ == "__main__":

    datamodule = CovidCT()

    vit_model = ViTModel()

    trainer = pl.Trainer(gpus=1, max_epochs=100)

    trainer.fit(vit_model, datamodule)
