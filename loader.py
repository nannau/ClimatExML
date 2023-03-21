import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class ClimatExMLLoader(Dataset):
    def __init__(self, var_glob) -> None:
        self.var_glob = var_glob

    def __len__(self):
        return len(self.var_glob[0])

    def __getitem__(self, idx):
        return torch.stack([torch.load(var[idx]) for var in self.var_glob], dim=0)


class ClimatExMLData(pl.LightningDataModule):
    def __init__(self, data_glob: dict, num_workers: int = 24, batch_size: int = 64):
        super().__init__()
        self.data_glob = data_glob
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.lr_test = ClimatExMLLoader(self.data_glob['lr_test'])
        self.hr_test = ClimatExMLLoader(self.data_glob['hr_test'])

        self.lr_train = ClimatExMLLoader(self.data_glob['lr_train'])
        self.hr_train = ClimatExMLLoader(self.data_glob['hr_train'])

    def train_dataloader(self):
        return {
            "lr": DataLoader(self.lr_train, batch_size=self.batch_size, num_workers=self.num_workers),
            "hr": DataLoader(self.hr_train, batch_size=self.batch_size, num_workers=self.num_workers)
        }

    def test_dataloader(self):
        return {
            "lr": DataLoader(self.lr_test, batch_size=self.batch_size),
            "hr": DataLoader(self.hr_test, batch_size=self.batch_size)
        }
