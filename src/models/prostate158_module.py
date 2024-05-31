from typing import Any, Dict, Tuple, Optional

import torch
from lightning import LightningModule
from monai.inferers import SlidingWindowInferer, SimpleInferer, Inferer
from monai.losses import DiceFocalLoss, DiceCELoss
from monai.metrics import DiceHelper, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.optimizers import Novograd
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score

from monai.networks.nets import UNet

from src.data.components.transforms import get_val_post_transforms, get_pre_metric_transforms_pred, \
    get_pre_metric_transforms_label


class Prostate158LitModule(LightningModule):
    def __init__(
            self,
            optimizer: Optional[torch.optim.Optimizer],
            scheduler: torch.optim.lr_scheduler,
            compile: bool = False,
            ndim: int = 3,
            prostate_mode: str = "anatomy",  # "anatomy", "tumor", "both"
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=4,
            act='PRELU',
            norm='BATCH',
            dropout=0.15,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net'])

        if prostate_mode == "anatomy":
            self.criterion = DiceCELoss(include_background=False,
                                        softmax=True,
                                        to_onehot_y=True)
            image_cols = ['t2']
            out_channels = 3
        elif prostate_mode == "tumor":
            self.criterion = DiceFocalLoss(include_background=False,
                                           softmax=True,
                                           to_onehot_y=True)
            image_cols = ['t2', 'adc', 'dwi']
            out_channels = 2
        elif prostate_mode == "both":
            self.criterion = DiceFocalLoss(include_background=False,
                                           softmax=True,
                                           to_onehot_y=True)
            image_cols = ['t2', 'adc', 'dwi']
            out_channels = 4
        else:
            raise ValueError("Prostate mode must be either 'anatomy' or 'tumor' or 'both'")

        self.net = UNet(
            spatial_dims=ndim,
            in_channels=len(image_cols),
            out_channels=out_channels,
            channels=list(channels),
            strides=list(strides),
            num_res_units=num_res_units,
            act=act,
            norm=norm,
            dropout=dropout,
        )

        self.val_inferer = SlidingWindowInferer(
            roi_size=(96, 96, 96),
            sw_batch_size=4,
            overlap=0.5
        )
        self.train_inferer = SimpleInferer()

        self.mean_dice_calculator = DiceHelper(
            include_background=False,
            reduction="mean",
        )
        self.hausdorff_distance_calculator = HausdorffDistanceMetric(
            include_background=False,
            reduction="mean",
        )
        self.surface_distance_calculator = SurfaceDistanceMetric(
            include_background=False,
            reduction="mean",
        )
        self.pre_metric_transform_pred = get_pre_metric_transforms_pred(out_channels)
        self.pre_metric_transform_label = get_pre_metric_transforms_label(out_channels)

        # metric objects for calculating and averaging accuracy across batches
        self.train_mean_dice_score = MeanMetric()
        self.train_hausdorff_distance_c0 = MeanMetric()
        self.train_hausdorff_distance_c1 = MeanMetric()
        self.train_hausdorff_distance_c2 = MeanMetric()
        self.train_hausdorff_distance_mean = MeanMetric()
        self.train_surface_distance_c0 = MeanMetric()
        self.train_surface_distance_c1 = MeanMetric()
        self.train_surface_distance_c2 = MeanMetric()
        self.train_surface_distance_mean = MeanMetric()

        self.val_mean_dice_score = MeanMetric()
        self.val_hausdorff_distance_c0 = MeanMetric()
        self.val_hausdorff_distance_c1 = MeanMetric()
        self.val_hausdorff_distance_c2 = MeanMetric()
        self.val_hausdorff_distance_mean = MeanMetric()
        self.val_surface_distance_c0 = MeanMetric()
        self.val_surface_distance_c1 = MeanMetric()
        self.val_surface_distance_c2 = MeanMetric()
        self.val_surface_distance_mean = MeanMetric()

        self.test_mean_dice_score = MeanMetric()
        self.test_hausdorff_distance_c0 = MeanMetric()
        self.test_hausdorff_distance_c1 = MeanMetric()
        self.test_hausdorff_distance_c2 = MeanMetric()
        self.test_hausdorff_distance_mean = MeanMetric()
        self.test_surface_distance_c0 = MeanMetric()
        self.test_surface_distance_c1 = MeanMetric()
        self.test_surface_distance_c2 = MeanMetric()
        self.test_surface_distance_mean = MeanMetric()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_mean_dice_score_best = MaxMetric()

    def forward(self,
                x: torch.Tensor,
                inferer: Inferer) -> torch.Tensor:
        return inferer(x, self.net)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

        self.val_mean_dice_score.reset()
        self.val_hausdorff_distance_c0.reset()
        self.val_hausdorff_distance_c1.reset()
        self.val_hausdorff_distance_c2.reset()
        self.val_hausdorff_distance_mean.reset()
        self.val_surface_distance_c0.reset()
        self.val_surface_distance_c1.reset()
        self.val_surface_distance_c2.reset()
        self.val_surface_distance_mean.reset()

        self.val_mean_dice_score_best.reset()

    def model_step(
            self,
            batch: Dict[str, torch.Tensor],
            is_train_step: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # batch - val - [2, 1, 217, 217, 139]
        # batch - output - [2, 3, 217, 217, 139]

        # batch - train [16, 1, 96, 96, 96]
        # batch - output [16, 3, 96, 96, 96]
        x, y = batch['image'], batch['label']
        preds = self.forward(x, self.train_inferer if is_train_step else self.val_inferer)
        # two cases
        # for train
        #             engine.state.output[Keys.LOSS] = engine.loss_function(engine.state.output[Keys.PRED], targets).mean()
        # for val
        #  eval_loss_handler = ignite.metrics.Loss(
        #             loss_fn=self.loss_function,
        #             output_transform=lambda output: (
        #                 output[0]['pred'].unsqueeze(0),  # add batch dim
        #                 output[0]['label'].argmax(0, keepdim=True).unsqueeze(0)  # reverse one-hot, add batch dim
        #             )
        #         )
        loss = self.criterion(preds, y)

        return loss, preds, y

    def training_step(
            self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        # computing metrics here takes long time, much memory and nan/inf values due to slicing issues (gt all 0 or 1's)
        # y = torch.stack([self.pre_metric_transform_label({'label': targets[i]})['label'] for i in range(len(targets))])
        # preds = torch.stack([self.pre_metric_transform_pred({'pred': preds[1]})['pred'] for i in range(len(targets))])
        # (mean_dice,
        #  not_nans) = self.mean_dice_calculator(y_pred=preds,
        #                                        y=y)
        # self.train_mean_dice_score(mean_dice.item())
        # del y, preds, targets

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/mean_dice", self.train_mean_dice_score, on_step=True, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        y = self.pre_metric_transform_label({'label': targets[0]})['label']
        preds = self.pre_metric_transform_pred({'pred': preds[0]})['pred']

        (mean_dice,
         not_nans) = self.mean_dice_calculator(y_pred=preds.unsqueeze(0),
                                               y=y.unsqueeze(0))
        hausdorff_score = self.hausdorff_distance_calculator(y_pred=preds.unsqueeze(0),
                                                             y=y.unsqueeze(0))  # [1, 2] => [B, C]
        surface_score = self.surface_distance_calculator(y_pred=preds.unsqueeze(0),
                                                         y=y.unsqueeze(0))  # [1, 2] => [B, C]
        # update and log metrics
        self.val_mean_dice_score(mean_dice.item())
        self.val_hausdorff_distance_c0(hausdorff_score[0, 0].item())
        if y.size(0) > 2:
            self.val_hausdorff_distance_c1(hausdorff_score[0, 1].item())
            self.log("val/hausdorff_c1", self.val_hausdorff_distance_c1, on_step=False, on_epoch=True, prog_bar=True)
        if y.size(0) > 3:
            self.val_hausdorff_distance_c2(hausdorff_score[0, 2].item())
            self.log("val/hausdorff_c2", self.val_hausdorff_distance_c2, on_step=False, on_epoch=True, prog_bar=True)
        self.val_hausdorff_distance_mean(hausdorff_score.mean().item())

        self.val_surface_distance_c0(surface_score[0, 0].item())
        if y.size(0) > 2:
            self.val_surface_distance_c1(surface_score[0, 1].item())
            self.log("val/surface_c1", self.val_surface_distance_c1, on_step=False, on_epoch=True, prog_bar=True)
        if y.size(0) > 3:
            self.val_surface_distance_c2(surface_score[0, 2].item())
            self.log("val/surface_c2", self.val_surface_distance_c2, on_step=False, on_epoch=True, prog_bar=True)

        self.val_surface_distance_mean(surface_score.mean().item())

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mean_dice", self.val_mean_dice_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/hausdorff_c0", self.val_hausdorff_distance_c0, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/hausdorff_mean", self.val_hausdorff_distance_mean, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/surface_c0", self.val_surface_distance_c0, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/surface_mean", self.val_surface_distance_mean, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        mean_dice = self.val_mean_dice_score.compute()  # get current val acc
        self.val_mean_dice_score_best(mean_dice)  # update best so far val acc

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/mean_dice_best", self.val_mean_dice_score_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        y = self.pre_metric_transform_label({'label': targets[0]})['label']
        preds = self.pre_metric_transform_pred({'pred': preds[0]})['pred']

        (mean_dice,
         not_nans) = self.mean_dice_calculator(y_pred=preds.unsqueeze(0),
                                               y=y.unsqueeze(0))
        hausdorff_score = self.hausdorff_distance_calculator(y_pred=preds.unsqueeze(0),
                                                             y=y.unsqueeze(0))  # [1, 2] => [B, C]
        surface_score = self.surface_distance_calculator(y_pred=preds.unsqueeze(0),
                                                         y=y.unsqueeze(0))  # [1, 2] => [B, C]
        # update and log metrics
        self.test_mean_dice_score(mean_dice.item())
        self.test_hausdorff_distance_c0(hausdorff_score[0, 0].item())
        if y.size(0) > 2:
            self.test_hausdorff_distance_c1(hausdorff_score[0, 1].item())
            self.log("test/hausdorff_c1", self.test_hausdorff_distance_c1, on_step=False, on_epoch=True, prog_bar=True)
        if y.size(0) > 3:
            self.test_hausdorff_distance_c2(hausdorff_score[0, 2].item())
            self.log("test/hausdorff_c2", self.test_hausdorff_distance_c2, on_step=False, on_epoch=True, prog_bar=True)
        self.test_hausdorff_distance_mean(hausdorff_score.mean().item())
        self.test_surface_distance_c0(surface_score[0, 0].item())
        if y.size(0) > 2:
            self.test_surface_distance_c1(surface_score[0, 1].item())
            self.log("test/surface_c1", self.test_surface_distance_c1, on_step=False, on_epoch=True, prog_bar=True)
        if y.size(0) > 3:
            self.test_surface_distance_c2(surface_score[0, 2].item())
            self.log("test/surface_c2", self.test_surface_distance_c2, on_step=False, on_epoch=True, prog_bar=True)
        self.test_surface_distance_mean(surface_score.mean().item())

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mean_dice", self.test_mean_dice_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/hausdorff_c0", self.test_hausdorff_distance_c0, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/hausdorff_mean", self.test_hausdorff_distance_mean, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/surface_c0", self.test_surface_distance_c0, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/surface_mean", self.test_surface_distance_mean, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def predict_step(self, batch: Any,
                     batch_idx: int,
                     dataloader_idx: int = 0) -> Any:
        x, y = batch['image'], batch['label']
        preds = self.forward(x)
        return preds, y

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        if self.hparams.optimizer is not None:
            optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        else:
            optimizer = Novograd(params=self.trainer.model.parameters(),
                                 lr=0.001,
                                 weight_decay=0.01,
                                 amsgrad=True)

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # "monitor": "val/loss",
                    "interval": "step",  # "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}


if __name__ == "__main__":
    print('ok')
