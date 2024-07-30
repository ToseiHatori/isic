from logging import getLogger
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from sklearn.metrics import auc, average_precision_score, roc_auc_score, roc_curve
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader

from run.init.dataset import init_datasets_from_config
from run.init.forwarder import Forwarder
from run.init.model import init_model_from_config
from run.init.optimizer import init_optimizer_from_config
from run.init.preprocessing import Preprocessing
from run.init.scheduler import init_scheduler_from_config
from src.datasets.wrapper import WrapperDataset

logger = getLogger(__name__)


def pAUCscore(
    label: np.ndarray, prediction: np.ndarray, min_tpr: float = 0.80
) -> float:
    v_gt = abs(label - 1)
    v_pred = np.array([1.0 - x for x in prediction])
    max_fpr = abs(1 - min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
    # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
    partial_auc = 0.5 * max_fpr ** 2 + (max_fpr - 0.5 * max_fpr ** 2) / (1.0 - 0.5) * (
        partial_auc_scaled - 0.5
    )

    return partial_auc


def pf_score(labels, predictions, percentile=0, bin=False):
    beta = 1
    y_true_count = 0
    ctp = 0
    cfp = 0

    predictions = predictions.copy()
    th = np.percentile(predictions, percentile)
    predictions[np.where(predictions < th)] = 0
    if bin:
        predictions[np.where(predictions >= th)] = 1

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if labels[idx]:
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    if y_true_count:
        c_recall = ctp / y_true_count
    else:
        c_recall = 0
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
        return result
    else:
        return 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class PLModel(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg.copy()
        self.save_embed = self.cfg.training.save_embed
        self.accelerator = cfg.training.accelerator
        pretrained = False if cfg.training.debug else True
        model = init_model_from_config(cfg.model, pretrained=pretrained)
        self.forwarder = Forwarder(cfg.forwarder, model)

        raw_datasets = init_datasets_from_config(cfg.dataset)
        self.val_outputs = []
        self.test_outputs = []

        preprocessing = Preprocessing(cfg.augmentation, **cfg.preprocessing)
        self.datasets = {}
        transforms = {
            "train": preprocessing.get_train_transform(),
            "val": preprocessing.get_val_transform(),
            "test": preprocessing.get_test_transform(),
        }
        for phase in ["train", "val", "test"]:
            if phase == "train":
                train_dataset = WrapperDataset(
                    raw_datasets["train"],
                    transforms["train"],
                    "train",
                    view=cfg.dataset.view,
                    lat=cfg.dataset.lat,
                )
                pos_cnt = train_dataset.base.df["target"].sum() * (
                    cfg.dataset.positive_aug_num + 1
                )
                pos_has_lesion_id_cnt = train_dataset.base.df["has_lesion_id"].sum()
                if cfg.dataset.positive_aug_num > 0:
                    train_positive_dataset = WrapperDataset(
                        raw_datasets["train_positive"],
                        transforms["train"],
                        "train",
                        view=cfg.dataset.view,
                    )
                    train_dataset = [train_dataset] + [
                        train_positive_dataset
                        for _ in range(cfg.dataset.positive_aug_num)
                    ]
                    train_dataset = ConcatDataset(train_dataset)
                self.datasets["train"] = train_dataset
                logger.info(f"{phase}: {len(self.datasets[phase])}")
                logger.info(f"{phase} positive records: {pos_cnt}")
                logger.info(
                    f"{phase} positive lesion_id records: {pos_has_lesion_id_cnt}"
                )
            else:
                self.datasets[phase] = WrapperDataset(
                    raw_datasets[phase], transforms[phase], phase
                )
                pos_cnt = self.datasets[phase].base.df["target"].sum()
                pos_has_lesion_id_cnt = (
                    self.datasets[phase].base.df["has_lesion_id"].sum()
                )
                logger.info(f"{phase}: {len(self.datasets[phase])}")
                logger.info(f"{phase} positive records: {pos_cnt}")
                logger.info(
                    f"{phase} positive lesion_id records: {pos_has_lesion_id_cnt}"
                )

        logger.info(
            f"training steps per epoch: {len(self.datasets['train'])/cfg.training.batch_size}"
        )
        self.cfg.scheduler.num_steps_per_epoch = (
            len(self.datasets["train"]) / cfg.training.batch_size
        )

    def on_train_epoch_start(self):
        pass

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        additional_info = {}
        (
            _,
            loss,
            loss_target,
            loss_age_scaled,
            loss_sex_enc,
            loss_anatom_site_general_enc,
            loss_has_lesion_id,
            loss_tbp_lv_H,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self.forwarder.forward(
            batch, phase="train", epoch=self.current_epoch, **additional_info
        )

        self.log(
            "train_loss",
            loss.detach().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            # logger=True,
            sync_dist=True,
        )
        self.log(
            "train_loss_target",
            loss_target.detach().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            # logger=True,
            sync_dist=True,
        )
        self.log(
            "train_loss_age_scaled",
            loss_age_scaled.detach().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            # logger=True,
            sync_dist=True,
        )
        self.log(
            "train_loss_sex_enc",
            loss_sex_enc.detach().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            # logger=True,
            sync_dist=True,
        )
        self.log(
            "train_loss_anatom_site_general_enc",
            loss_anatom_site_general_enc.detach().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            # logger=True,
            sync_dist=True,
        )
        self.log(
            "train_loss_has_lesion_id",
            loss_has_lesion_id.detach().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            # logger=True,
            sync_dist=True,
        )
        self.log(
            "train_loss_tbp_lv_H",
            loss_tbp_lv_H.detach().item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            # logger=True,
            sync_dist=True,
        )
        sch = self.lr_schedulers()
        sch.step()
        self.log(
            "lr",
            sch.get_last_lr()[0],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            # logger=True,
            sync_dist=True,
            batch_size=1,
        )
        self.log(
            "lr_head",
            sch.get_last_lr()[1],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            # logger=True,
            sync_dist=True,
            batch_size=1,
        )

        return loss

    def _end_process(self, outputs: List[Dict[str, Tensor]], phase: str):
        # Aggregate results
        epoch_results: Dict[str, np.ndarray] = {}
        outputs = self.all_gather(outputs)
        """
        # デバッグ用
        for i, gpu_output in enumerate(outputs):
            print(f"{i} output sizes:")
            for key in gpu_output:
                if isinstance(gpu_output[key], Tensor):
                    print(f"  {key}: {gpu_output[key].size()}")
                else:
                    print(f"  {key}: {len(gpu_output[key])}")
        """

        for key in [
            "original_index",
            "isic_id",
            "patient_id",
            "label",
            "pred",
            "pred_age_scaled",
            "pred_sex_enc",
            "pred_anatom_site_general_enc",
            "pred_has_lesion_id",
            "pred_tbp_lv_H",
            "embed_features",
        ]:
            if key == "embed_features":
                if not self.save_embed:
                    continue
            if isinstance(outputs[0][key], Tensor):
                result = torch.cat([torch.atleast_1d(x[key]) for x in outputs], dim=1)
                result = torch.flatten(result, end_dim=1)
                epoch_results[key] = result.detach().cpu().numpy()
            else:
                result = np.concatenate([x[key] for x in outputs])
                """
                # DDP用のコード（なんかうまく動いていない）
                if key == "isic_id":
                    result = ["ISIC_" + str(x).zfill(7) for x in result]
                elif key == "patient_id":
                    result = ["IP_" + str(x).zfill(7) for x in result]
                """
                epoch_results[key] = result
        df = pd.DataFrame(
            data={
                "original_index": epoch_results["original_index"]
                .reshape(-1)
                .astype(int),
                "isic_id": epoch_results["isic_id"].reshape(-1),
                "patient_id": epoch_results["patient_id"].reshape(-1),
            }
        )

        df["pred"] = sigmoid(
            epoch_results["pred"][:, 0].reshape(-1).astype(np.float128)
        )
        df["label"] = epoch_results["label"]
        df["pred_age_scaled"] = (
            sigmoid(
                epoch_results["pred_age_scaled"][:, 0].reshape(-1).astype(np.float128)
            )
            * 90
        )
        df["pred_sex_enc"] = sigmoid(
            epoch_results["pred_sex_enc"][:, 0].reshape(-1).astype(np.float128)
        )
        df["pred_anatom_site_general_enc"] = (
            epoch_results["pred_anatom_site_general_enc"].argmax(axis=1).reshape(-1)
        )
        df["pred_has_lesion_id"] = sigmoid(
            epoch_results["pred_has_lesion_id"][:, 0].reshape(-1).astype(np.float128)
        )
        df["pred_tbp_lv_H"] = (
            sigmoid(
                epoch_results["pred_tbp_lv_H"][:, 0].reshape(-1).astype(np.float128)
            )
            * 110
        ) + 2
        df = df.sort_values(by="original_index")

        if phase == "test" and self.trainer.global_rank == 0:
            # Save test results ".npz" format
            test_results_filepath = Path(self.cfg.out_dir) / "test_results"
            if not test_results_filepath.exists():
                test_results_filepath.mkdir(exist_ok=True)
            np.savez_compressed(
                str(test_results_filepath / "test_results.npz"),
                **epoch_results,
            )
            df.to_csv(test_results_filepath / "test_results.csv", index=False)
            if self.datasets[phase].base.data_name == "vindr":
                df_vindr = pd.read_csv("./vindr/vindr_train.csv")
                df_vindr["cancer"] = df["pred"]
                df_vindr.to_csv(test_results_filepath / "vinder_pl.csv", index=False)

        if phase != "test" and self.trainer.global_rank == 0:
            test_results_filepath = Path(self.cfg.out_dir) / "test_results"
            if not test_results_filepath.exists():
                test_results_filepath.mkdir(exist_ok=True)
            df.to_csv(
                test_results_filepath / f"epoch_{self.current_epoch}_results.csv",
                index=False,
            )
            weights_filepath = Path(self.cfg.out_dir) / "weights"
            if not weights_filepath.exists():
                weights_filepath.mkdir(exist_ok=True)
            weights_path = str(
                weights_filepath / f"model_weights_epoch_{self.current_epoch}.pth"
            )
            # logger.info(f"Extracting and saving weights: {weights_path}")
            torch.save(self.forwarder.model.state_dict(), weights_path)

        pred = df["pred"].values
        label = df["label"].values
        pf_score_000 = pf_score(label, pred)
        try:
            auc_score = roc_auc_score(label.reshape(-1), pred.reshape(-1))
        except Exception:
            auc_score = 0
        try:
            pauc_score = pAUCscore(label.reshape(-1), pred.reshape(-1))
        except Exception as e:
            logger.error(e)
            pauc_score = 0
        try:
            pr_auc_score = average_precision_score(label.reshape(-1), pred.reshape(-1))
        except Exception:
            auc_score = 0

        mean_auc_score = (auc_score + pr_auc_score + pauc_score) / 3

        # Log items
        for loss_name in [
            "loss",
            "loss_target",
            "loss_age_scaled",
            "loss_sex_enc",
            "loss_anatom_site_general_enc",
            "loss_has_lesion_id",
            "loss_tbp_lv_H",
        ]:
            tmp_loss = (
                torch.cat([torch.atleast_1d(x[loss_name]) for x in outputs])
                .detach()
                .cpu()
                .numpy()
            )
            tmp_mean_loss = np.mean(tmp_loss)
            self.log(
                f"{phase}/{loss_name}", tmp_mean_loss, prog_bar=True, sync_dist=True
            )
        self.log(f"{phase}/pAUC", pauc_score, prog_bar=True, sync_dist=True)
        self.log(f"{phase}/pf_score", pf_score_000, prog_bar=False, sync_dist=True)
        self.log(f"{phase}/pr_auc", pr_auc_score, prog_bar=True, sync_dist=True)
        self.log(f"{phase}/auc", auc_score, prog_bar=True, sync_dist=True)
        self.log(f"{phase}/mean_auc", mean_auc_score, prog_bar=True, sync_dist=True)

    def _evaluation_step(self, batch: Dict[str, Tensor], phase: Literal["val", "test"]):
        (
            preds,
            loss,
            loss_target,
            loss_age_scaled,
            loss_sex_enc,
            loss_anatom_site_general_enc,
            loss_has_lesion_id,
            loss_tbp_lv_H,
            embed_features,
            preds_age_scaled,
            preds_sex_enc,
            preds_anatom_site_general_enc,
            preds_has_lesion_id,
            preds_tbp_lv_H,
        ) = self.forwarder.forward(batch, phase=phase, epoch=self.current_epoch)

        output = {
            "loss": loss,
            "loss_target": loss_target,
            "loss_age_scaled": loss_age_scaled,
            "loss_sex_enc": loss_sex_enc,
            "loss_anatom_site_general_enc": loss_anatom_site_general_enc,
            "loss_has_lesion_id": loss_has_lesion_id,
            "loss_tbp_lv_H": loss_tbp_lv_H,
            "label": batch["label"],
            "original_index": batch["original_index"],
            "patient_id": batch["patient_id"],
            "isic_id": batch["isic_id"],
            "pred": preds.detach(),
            "pred_age_scaled": preds_age_scaled.detach(),
            "pred_sex_enc": preds_sex_enc.detach(),
            "pred_anatom_site_general_enc": preds_anatom_site_general_enc.detach(),
            "pred_has_lesion_id": preds_has_lesion_id.detach(),
            "pred_tbp_lv_H": preds_tbp_lv_H.detach(),
            "embed_features": embed_features.detach(),
        }
        return output

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):
        outputs = self._evaluation_step(batch, phase="val")
        self.val_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self) -> None:
        self._end_process(self.val_outputs, "val")
        self.val_outputs = []

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        outputs = self._evaluation_step(batch, phase="test")
        self.test_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self) -> None:
        self._end_process(self.test_outputs, "test")
        self.test_outputs = []

    def configure_optimizers(self):
        model = self.forwarder.model
        opt_cls, kwargs = init_optimizer_from_config(
            self.cfg.optimizer, model.forward_features.parameters()
        )

        self.cfg.optimizer.lr = self.cfg.optimizer.lr_head
        kwargs_head = init_optimizer_from_config(
            self.cfg.optimizer, model.head.parameters(), return_cls=False
        )

        optimizer = opt_cls([kwargs, kwargs_head])
        scheduler = init_scheduler_from_config(self.cfg.scheduler, optimizer)

        if scheduler is None:
            return [optimizer]
        return [optimizer], [scheduler]

    def on_before_zero_grad(self, *args, **kwargs):
        self.forwarder.ema.update(self.forwarder.model.parameters())

    def _dataloader(self, phase: str) -> DataLoader:
        logger.info(f"{phase} data loader called")
        dataset = self.datasets[phase]

        batch_size = self.cfg.training.batch_size
        num_workers = self.cfg.training.num_workers

        num_gpus = self.cfg.training.num_gpus
        if phase != "train":
            batch_size = self.cfg.training.batch_size_test
        batch_size //= num_gpus
        num_workers //= num_gpus

        drop_last = True if self.cfg.training.drop_last and phase == "train" else False
        shuffle = phase == "train"

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        return loader

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(phase="train")

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(phase="val")

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(phase="test")
