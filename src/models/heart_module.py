from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score

from src.models.components.unet import UNet


class HeartLitModule(LightningModule):
    def __init__(
            self,
            net: UNet,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net'])

        self.net = net

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_precision = BinaryPrecision(multidim_average="global")
        self.train_recall = BinaryRecall(multidim_average="global")
        self.train_f_score = BinaryF1Score(multidim_average="global")

        self.val_precision = BinaryPrecision(multidim_average="global")
        self.val_recall = BinaryRecall(multidim_average="global")
        self.val_f_score = BinaryF1Score(multidim_average="global")

        self.test_precision = BinaryPrecision(multidim_average="global")
        self.test_recall = BinaryRecall(multidim_average="global")
        self.test_f_score = BinaryF1Score(multidim_average="global")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_precision_best = MaxMetric()
        self.val_recall_best = MaxMetric()
        self.val_f_score_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f_score.reset()

        self.val_precision_best.reset()
        self.val_recall_best.reset()
        self.val_f_score_best.reset()

    def model_step(
            self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch['image'], batch['mask']
        logits = self.forward(x).squeeze(1)
        loss = self.criterion(logits, y.float())
        return loss, logits, y

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
        self.train_loss(loss)
        self.train_precision(preds, targets)
        self.train_recall(preds, targets)
        self.train_f_score(preds, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/precision", self.train_precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recall", self.train_recall, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/f_score", self.train_f_score, on_step=True, on_epoch=True, prog_bar=True)
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

        # update and log metrics
        self.val_loss(loss)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_f_score(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f_score", self.val_f_score, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_precision.compute()  # get current val acc
        self.val_precision_best(acc)  # update best so far val acc

        acc = self.val_recall.compute()  # get current val acc
        self.val_recall_best(acc)  # update best so far val acc

        acc = self.val_f_score.compute()  # get current val acc
        self.val_f_score_best(acc)  # update best so far val acc

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/precision_best", self.val_precision_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/recall_best", self.val_recall_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/f_score_best", self.val_f_score_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.test_f_score(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f_score", self.test_f_score, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def predict_step(self, batch: Any,
                     batch_idx: int,
                     dataloader_idx: int = 0) -> Any:
        x, img_ids = batch['image'], batch['img_id']
        gt = batch['mask']
        logits = self.forward(x).squeeze(1)
        return logits, img_ids, gt

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
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    print('ok')
