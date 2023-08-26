"""
MNIST Pytorch Lightning Data Module.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import os
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from typing import Optional, Union


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: Union[Path, str] = "./data",
        batch_size: int = 128,
        num_workers: int = os.cpu_count() // 2,
        seed: int = 42,
        val_partition: float = 1 / 12
    ):
        """
        Args:
            root: directory path to save the MNIST dataset to.
            batch_size: batch size. Default 128.
            num_workers: number of workers. Default half the CPU count.
            seed: random seed. Default 42.
            val_partition: proportion of training dataset to use for
                validation Default 1 / 12.
        """
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.val_partition = min(max(val_partition, 0.0), 1.0)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        self.x_dims = (1, 28, 28)
        self.num_labels = 10

    def prepare_data(self) -> None:
        """
        Download MNIST dataset on a single CPU process.
        Input:
            None.
        """
        _ = MNIST(
            self.root, train=True, download=True, transform=self.transform
        )
        _ = MNIST(
            self.root, train=False, download=True, transform=self.transform
        )
        return

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Split dataset into train, val, and test partitions.
        Input:
            stage: setup stage. One of [`fit`, `test`, None]. Default None.
        Returns:
            None.
        """
        self.train, self.val, self.test = None, None, None

        if stage is None or stage == "fit":
            full = MNIST(self.root, train=True, transform=self.transform)
            if 0.0 < self.val_partition < 1.0:
                partition = [
                    int((1.0 - self.val_partition) * len(full)),
                    int(self.val_partition * len(full))
                ]
                self.train, self.val = random_split(full, partition)
            else:
                self.train, self.val = full, None

        if stage is None or stage == "test":
            self.test = MNIST(self.root, train=False, transform=self.transform)

        return

    def train_dataloader(self) -> Optional[DataLoader]:
        """
        Returns the training dataloader.
        Input:
            None.
        Returns:
            Training dataloader.
        """
        if self.train is None:
            return None
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """
        Returns the validation dataloader.
        Input:
            None.
        Returns:
            Validation dataloader.
        """
        if self.val is None:
            return None
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """
        Returns the test dataloader.
        Input:
            None.
        Returns:
            Test dataloader.
        """
        if self.test is None:
            return None
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
