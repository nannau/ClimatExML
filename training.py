import pytorch_lightning as pl
import torch.nn.functional as F
import torch


class SuperModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        lr, hr = batch["lr"], batch["hr"]
        sr = self.model(lr)
        loss = F.mse_loss(sr, hr)
        self.log("Train MSE", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.00025,  betas=(0.9, 0.99))

    def test_step(self, batch, batch_idx):
        lr, hr = batch["lr"], batch["hr"]
        sr = self.model(lr)
        loss = F.mse_loss(sr, hr)
        self.log("Test MSE", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
