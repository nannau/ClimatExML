import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class ClimatExMLLoader(Dataset):
    def __init__(self, lr_glob, hr_glob) -> None:
        self.lr_glob = lr_glob
        self.hr_glob = hr_glob

    def __len__(self):
        return len(self.lr_glob[0])

    def __getitem__(self, idx):
        lr = torch.stack([torch.load(var[idx]) for var in self.lr_glob], dim=1)
        hr = torch.stack([torch.load(var[idx]) for var in self.hr_glob], dim=1)

        return [lr, hr]


class ClimatExMLData(pl.LightningDataModule):
    def __init__(
        self, data_glob: dict = None, num_workers: int = 24, batch_size: int = None
    ):
        super().__init__()
        self.data_glob = data_glob
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        # self.test_data = ClimatExMLLoader(
        #     self.data_glob["lr_test"], self.data_glob["hr_test"]
        # )
        self.train_data = ClimatExMLLoader(
            self.data_glob["lr_train"], self.data_glob["hr_train"]
        )

    def train_dataloader(self):
        return (
            DataLoader(
                self.train_data,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            ),
        )

    def test_dataloader(self):
        # For some reason this can't be a dictionary?
        return (
            DataLoader(
                self.test_data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=True,
            ),
        )
