import torch
from torch.nn import functional as F
import pytorch_lightning as pl

from vit_pytorch import ViT
from experiments.Dataset import CovidCT


class ViTModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.vit = ViT(image_size=256, patch_size=32, num_classes=3, dim=1024, depth=6, heads=8, mlp_dim=2048,
                       dropout=0.1, emb_dropout=0.1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":

    datamodule = CovidCT()

    # init model
    vit_model = ViTModel()

    # Initialize a trainer
    trainer = pl.Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=20)

    # Train the model ⚡
    trainer.fit(vit_model, datamodule)
