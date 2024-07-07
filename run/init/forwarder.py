from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from global_objectives.losses import AUCPRLoss
from omegaconf import DictConfig
from torch import Tensor
from torch_ema import ExponentialMovingAverage


class Forwarder(nn.Module):
    def __init__(self, cfg: DictConfig, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        # workaround for device inconsistency of ExponentialMovingAverage
        self.ema = None
        self.cfg = cfg
        self.pr_auc_loss = AUCPRLoss()

    def loss_pr_auc(self, logits, labels):
        return 1 + 10 * self.pr_auc_loss(logits, labels)

    def loss_bce(
        self,
        logits,
        labels,
        mean=True,
    ) -> Tensor:

        loss = F.binary_cross_entropy_with_logits(
            logits.view(-1, 1), labels.view(-1, 1), reduction="none"
        )  # (B, 1)

        if mean:
            return torch.mean(loss)
        else:
            return loss

    def loss_ce(
        self,
        logits,
        labels,
        mean=True,
    ) -> Tensor:

        loss = F.cross_entropy(logits, labels, reduction="none")  # (B, C)

        if mean:
            return torch.mean(loss)
        else:
            return loss

    def loss(
        self,
        logits,
        logits_age_scaled,
        logits_sex_enc,
        logits_anatom_site_general_enc,
        labels,
        labels_age_scaled,
        labels_sex_enc,
        labels_anatom_site_general_enc,
    ):
        cfg = self.cfg.loss
        loss = self.loss_bce(logits, labels) * cfg.target_weight
        loss += (
            self.loss_bce(logits_age_scaled, labels_age_scaled) * cfg.age_scaled_weight
        )
        loss += self.loss_bce(logits_sex_enc, labels_sex_enc) * cfg.sex_enc_weight
        loss += (
            self.loss_ce(logits_anatom_site_general_enc, labels_anatom_site_general_enc)
            * cfg.anatom_site_general_enc_weight
        )
        return loss

    def forward(
        self, batch: Dict[str, Tensor], phase: str, epoch=None
    ) -> Tuple[Tensor, Tensor]:

        # workaround for device inconsistency of ExponentialMovingAverage
        if self.ema is None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.999)

        use_multi_view = self.cfg.use_multi_view
        use_multi_lat = self.cfg.use_multi_lat

        # inputs: Input tensor.
        inputs = batch["image"]

        if use_multi_view or use_multi_lat:
            bs, ch, h, w = inputs.shape
            inputs = inputs.view(bs * ch, 1, h, w)

        # labels
        labels = batch["label"].to(torch.float16)
        labels_age_scaled = batch["age_scaled"].to(torch.float16)
        labels_sex_enc = batch["sex_enc"].to(torch.float16)
        labels_anatom_site_general_enc = batch["anatom_site_general_enc"].to(
            torch.float16
        )

        if phase == "train":
            with torch.set_grad_enabled(True):
                embed_features = self.model.forward_features(inputs)
                logits = self.model.head.head(embed_features)
                logits_age_scaled = self.model.head.head_age_scaled(embed_features)
                logits_sex_enc = self.model.head.head_sex_enc(embed_features)
                logits_anatom_site_general_enc = (
                    self.model.head.head_anatom_site_general_enc(embed_features)
                )
        else:
            if phase == "test":
                with self.ema.average_parameters():
                    embed_features = self.model.forward_features(inputs)
                    logits = self.model.head.head(embed_features)
                    logits_age_scaled = self.model.head.head_age_scaled(embed_features)
                    logits_sex_enc = self.model.head.head_sex_enc(embed_features)
                    logits_anatom_site_general_enc = (
                        self.model.head.head_anatom_site_general_enc(embed_features)
                    )
            elif phase == "val":
                embed_features = self.model.forward_features(inputs)
                if use_multi_view:
                    embed_features = embed_features.view(bs, -1)
                logits = self.model.head.head(embed_features)
                logits_age_scaled = self.model.head.head_age_scaled(embed_features)
                logits_sex_enc = self.model.head.head_sex_enc(embed_features)
                logits_anatom_site_general_enc = (
                    self.model.head.head_anatom_site_general_enc(embed_features)
                )

        loss = self.loss(
            logits=logits,
            logits_age_scaled=logits_age_scaled,
            logits_sex_enc=logits_sex_enc,
            logits_anatom_site_general_enc=logits_anatom_site_general_enc,
            labels=labels,
            labels_age_scaled=labels_age_scaled,
            labels_sex_enc=labels_sex_enc,
            labels_anatom_site_general_enc=labels_anatom_site_general_enc,
        )
        return (
            logits,
            loss,
            embed_features,
            logits_age_scaled,
            logits_sex_enc,
            logits_anatom_site_general_enc,
        )
