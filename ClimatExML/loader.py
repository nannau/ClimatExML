import torch
from torch.utils.data import Dataset
import lightning as pl
from torch.utils.data import DataLoader


class ClimatExSampler(Dataset):
    lr_paths: list
    hr_paths: list
    hr_invariant_paths: list
    lr_invariant_paths: list

    def __init__(
        self, lr_paths, hr_paths, hr_invariant_paths, lr_invariant_paths
    ) -> None:
        super().__init__()
        self.lr_paths = lr_paths
        self.hr_paths = hr_paths
        self.hr_invariant_paths = hr_invariant_paths
        self.lr_invariant_paths = lr_invariant_paths

        self.hr_invariant = torch.stack([torch.load(path).float() for path in self.hr_invariant_paths])
        self.lr_invariant = torch.stack([torch.load(path).float() for path in self.lr_invariant_paths])

    def __len__(self):
        return len(self.lr_paths[0])

    def __getitem__(self, idx):
        lr = torch.stack([torch.load(var[idx]) for var in self.lr_paths], dim=1)
        lr = torch.cat([lr, self.lr_invariant.expand(lr.size(0), -1, -1, -1)], dim=1)

        hr = torch.stack([torch.load(var[idx]) for var in self.hr_paths], dim=1)
        hr_invariant_expand = self.hr_invariant.expand(hr.size(0), -1, -1, -1)

        return (lr, hr, hr_invariant_expand)


class ClimatExLightning(pl.LightningDataModule):
    def __init__(
        self,
        train_data,
        test_data,
        invariant,
        num_workers: int = 24,
    ):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.invariant = invariant
        self.num_workers = num_workers
        self.batch_size = 1

    def setup(
        self,
        stage=None,
    ):
        self.train_data = ClimatExSampler(
            self.train_data.lr_files,
            self.train_data.hr_files,
            self.invariant.hr_invariant_paths,
            self.invariant.lr_invariant_paths,
        )
        self.test_data = ClimatExSampler(
            self.test_data.lr_files,
            self.test_data.hr_files,
            self.invariant.hr_invariant_paths,
            self.invariant.lr_invariant_paths,
        )

    def train_dataloader(self):
        return (
            DataLoader(
                self.train_data,
                batch_size=1,
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
