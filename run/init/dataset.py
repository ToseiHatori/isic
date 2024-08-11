from typing import Dict

import pandas as pd
from omegaconf import DictConfig

from src.datasets.isic_dataset import ISICDataset


def init_datasets_from_config(cfg: DictConfig):
    if cfg.type == "isic":
        datasets = get_isic_dataset(
            num_folds=cfg.num_folds,
            test_fold=cfg.test_fold,
            val_fold=cfg.val_fold,
            seed=cfg.seed,
            num_records=cfg.num_records,
            phase=cfg.phase,
            cfg=cfg,
        )
    else:
        raise ValueError(f"Unknown dataset type: {cfg.type}")

    return datasets


def get_isic_dataset(
    num_folds: int,
    test_fold: int,
    val_fold: int,
    seed: int = 2023,
    num_records: int = 0,
    phase: str = "train",
    cfg=None,
) -> Dict[str, ISICDataset]:

    df = ISICDataset.create_dataframe(
        num_folds,
        seed,
        num_records,
        fold_path=cfg.fold_path,
        past_fold_path=cfg.past_fold_path,
        validation_on_past_data=cfg.validation_on_past_data,
    )

    train_df = df[(df["fold"] != val_fold) & (df["fold"] != test_fold)]
    val_df = df[df["fold"] == val_fold]
    test_df = df[df["fold"] == test_fold]
    if phase == "train":
        train_positive = train_df[train_df["target"] == 1]
        # ダウンサンプリング
        if cfg.downsampling_rate > 0:
            train_negative = train_df[train_df["target"] == 0]
            # positiveサンプルの数を取得
            positive_count = len(train_positive)
            # negativeサンプルをランダムにサンプリングして、positiveサンプルのn倍の数を取得
            negative_sample = train_negative.sample(
                n=positive_count * cfg.downsampling_rate, random_state=42
            )
            # positiveとサンプリングしたnegativeを結合
            train_balanced = pd.concat([train_positive, negative_sample])
            # データフレームをシャッフル
            train_df = train_balanced.sample(frac=1, random_state=42).reset_index(
                drop=True
            )
        if cfg.downsampling_by_patient:
            target_one_patient_ids = train_df[train_df["target"] == 1][
                "patient_id"
            ].unique()
            train_df = train_df[
                train_df["patient_id"].isin(target_one_patient_ids)
            ].reset_index(drop=True)

        train_dataset = ISICDataset(train_df, phase="train", cfg=cfg)
        val_dataset = ISICDataset(val_df, phase="test", cfg=cfg)
        test_dataset = ISICDataset(test_df, phase="test", cfg=cfg)
        train_positive_dataset = ISICDataset(train_positive, phase="train", cfg=cfg)
    elif phase == "valid":
        train_dataset = ISICDataset(train_df, phase="val", cfg=cfg)
        val_dataset = ISICDataset(train_df, phase="val", cfg=cfg)
        test_dataset = ISICDataset(train_df, phase="val", cfg=cfg)
    elif phase == "test":
        train_dataset = ISICDataset(test_df, phase="test", cfg=cfg)
        val_dataset = ISICDataset(test_df, phase="test", cfg=cfg)
        test_dataset = ISICDataset(test_df, phase="test", cfg=cfg)

    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    if phase == "train":
        datasets["train_positive"] = train_positive_dataset
    return datasets
