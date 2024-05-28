import os
from typing import Any, Dict, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.components.heart_dataset import HeartDataset


class HeartDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/",
            img_dir: str = "heart/images",
            label_path: str = "heart/golds",
            train_img_alias: str = "tr",
            test_img_alias: str = "ts",
            val_img_alias: str = "val",
            do_input_normalization: bool = True,
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_predict: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            if self.hparams.do_input_normalization:
                train_transformations = A.Compose([
                    A.Resize(512, 512),
                    A.ShiftScaleRotate(shift_limit=0.1,
                                       scale_limit=0.1,
                                       rotate_limit=30,
                                       p=0.5),
                    A.HorizontalFlip(),
                    A.Normalize(mean=0.5,
                                std=0.5),
                    ToTensorV2(),
                ])
            else:
                train_transformations = A.Compose([
                    A.Resize(512, 512),
                    A.ShiftScaleRotate(shift_limit=0.1,
                                       scale_limit=0.1,
                                       rotate_limit=30,
                                       p=0.5),
                    A.HorizontalFlip(),
                    ToTensorV2(),
                ])

            if self.hparams.do_input_normalization:
                pred_transformations = A.Compose([
                    A.Resize(512, 512),
                    A.Normalize(mean=0.5, std=0.5),
                    ToTensorV2(),
                ])
            else:
                pred_transformations = A.Compose([
                    A.Resize(512, 512),
                    ToTensorV2(),
                ])

            label_path = os.path.join(self.hparams.data_dir, self.hparams.label_path)
            img_dir = os.path.join(self.hparams.data_dir, self.hparams.img_dir)

            self.data_train = HeartDataset(
                label_path=label_path,
                img_dir=img_dir,
                img_alias=self.hparams.train_img_alias,
                transform=train_transformations,)

            self.data_val = HeartDataset(
                label_path=label_path,
                img_dir=img_dir,
                img_alias=self.hparams.val_img_alias,
                transform=pred_transformations)

            self.data_test = HeartDataset(
                label_path=label_path,
                img_dir=img_dir,
                img_alias=self.hparams.test_img_alias,
                transform=pred_transformations)

            self.data_predict = HeartDataset(
                label_path=label_path,
                img_dir=img_dir,
                img_alias=self.hparams.test_img_alias,
                transform=pred_transformations)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    data_dir = "/home/gsoykan/Desktop/ku/spring24_comp548_medical_imaging/hw3/unet_segmentation/data"
    datamodule = HeartDataModule(
        data_dir=data_dir)
    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    for batch in tqdm(dataloader):
        print(batch)
