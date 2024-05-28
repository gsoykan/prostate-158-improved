import os
import shutil
from functools import partial
from typing import Any, Dict, Optional, List, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from monai.engines import default_prepare_batch
from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm import tqdm

from src.data.components.heart_dataset import HeartDataset
from src.data.components.transform_config import get_anatomy_transform_config, get_tumor_transform_config, \
    get_both_transform_config
from src.data.components.transforms import get_train_transforms, get_val_transforms, get_test_transforms


class Prostate158DataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/",
            train_csv: str = "prostate158/train.csv",
            valid_csv: str = "prostate158/valid.csv",
            test_csv: str = "prostate158/test.csv",
            samples_root_dir: str = "prostate158",
            anatomy_image_cols: Tuple[str] = ("t2",),
            anatomy_label_cols: Tuple[str] = ("t2_anatomy_reader1",),
            tumor_image_cols: Tuple[str] = ("t2", "adc", "dwi"),
            tumor_label_cols: Tuple[str] = ("adc_tumor_reader1",),
            dataset_type: str = "persistent",
            dataset_mode: str = "anatomy",  # "anatomy", "tumor", "both"
            cache_dir: str = "/tmp/monai-cache",
            use_val_for_test: bool = True,
            batch_size: int = 2,
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

            train_df = pd.read_csv(str(os.path.join(self.hparams.data_dir, self.hparams.train_csv)))
            valid_df = pd.read_csv(str(os.path.join(self.hparams.data_dir, self.hparams.valid_csv)))
            test_df = pd.read_csv(str(os.path.join(self.hparams.data_dir, self.hparams.test_csv)))

            train_df['split'] = 'train'
            valid_df['split'] = 'valid'
            test_df['split'] = 'test'

            whole_df = []
            whole_df += [train_df]
            whole_df += [valid_df]
            whole_df += [test_df]
            df = pd.concat(whole_df)

            if self.hparams.dataset_mode == "anatomy":
                cols = list(self.hparams.anatomy_image_cols + self.hparams.anatomy_label_cols)
            elif self.hparams.dataset_mode == "tumor":
                cols = list(self.hparams.tumor_image_cols + self.hparams.tumor_label_cols)
            elif self.hparams.dataset_mode == "both":
                cols = list(set(self.hparams.anatomy_image_cols +
                                self.hparams.tumor_image_cols +
                                self.hparams.anatomy_label_cols +
                                self.hparams.tumor_label_cols))

            for col in cols:
                # create absolute file name from relative fn in df and data_dir
                df[col] = [os.path.join(self.hparams.data_dir,
                                        self.hparams.samples_root_dir,
                                        fn) for fn in df[col]]
                if not os.path.exists(list(df[col])[0]):
                    raise FileNotFoundError(list(df[col])[0])

            # data_dict is not the correct name,
            # list_of_data_dicts would be more accurate, but also longer.
            data_dict = [dict(row[1]) for row in df[cols].iterrows()]

            train_files = list(map(data_dict.__getitem__, *np.where(df.split == 'train')))
            val_files = list(map(data_dict.__getitem__, *np.where(df.split == 'valid')))
            if self.hparams.use_val_for_test:
                test_files = val_files
            else:
                test_files = list(map(data_dict.__getitem__, *np.where(df.split == 'test')))

            if self.hparams.dataset_mode == "anatomy":
                transform_config = get_anatomy_transform_config()
            elif self.hparams.dataset_mode == "tumor":
                transform_config = get_tumor_transform_config()
            elif self.hparams.dataset_mode == "both":
                transform_config = get_both_transform_config()

            train_transforms = get_train_transforms(transform_config)
            val_transforms = get_val_transforms(transform_config)
            test_transforms = get_test_transforms(transform_config)

            MonaiDataset = (Prostate158DataModule
                            .import_monai_dataset(self.hparams.dataset_type,
                                                  self.hparams.cache_dir))

            self.data_train = MonaiDataset(
                data=train_files,
                transform=train_transforms
            )
            self.data_val = MonaiDataset(
                data=val_files,
                transform=val_transforms
            )
            self.data_test = MonaiDataset(
                data=test_files,
                transform=test_transforms
            )
            self.data_predict = MonaiDataset(
                data=test_files,
                transform=test_transforms
            )

    @staticmethod
    def import_monai_dataset(dataset_type: str,
                             cache_dir: str):
        if dataset_type == 'persistent':
            from monai.data import PersistentDataset
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)  # rm previous cache DS
            os.makedirs(cache_dir, exist_ok=True)
            MonaiDataset = partial(PersistentDataset, cache_dir=cache_dir)
        elif dataset_type == 'cache':
            from monai.data import CacheDataset
            raise NotImplementedError('CacheDataset not yet implemented')
        else:
            from monai.data import Dataset as MonaiDataset
        return MonaiDataset

    def monai_collate(self, batch: List[Any]) -> Tuple[Any, ...]:
        collateds = []
        for batch_instance in batch:
            collateds.append(default_collate(batch_instance))
        result = {}
        for k in collateds[0].keys():
            result[k] = torch.cat(list(map(lambda x: x[k], collateds)), dim=0)
        # TODO: @gsoykan - maybe we can delete some keys here?
        # TODO: @gsoykan - in both case maybe merge labels in a sensible way...
        return result

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.monai_collate
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.monai_collate
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.monai_collate
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.monai_collate
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
    data_dir = "/home/gsoykan/Desktop/ku/spring24_comp548_medical_imaging/term_project/unet_segmentation/data"
    datamodule = Prostate158DataModule(
        data_dir=data_dir,
        batch_size=2,
        dataset_mode="both")
    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    for batch in tqdm(dataloader):
        print(batch)
