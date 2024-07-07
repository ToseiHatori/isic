from typing import Dict

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
    )

    train_df = df[(df["fold"] != val_fold) & (df["fold"] != test_fold)]
    val_df = df[df["fold"] == val_fold]
    test_df = df[df["fold"] == test_fold]
    if phase == "train":
        train_positive = train_df[train_df["target"] == 1]

        train_dataset = ISICDataset(train_df, phase="train", cfg=cfg)
        val_dataset = ISICDataset(val_df, phase="test", cfg=cfg)
        test_dataset = ISICDataset(test_df, phase="test", cfg=cfg)
        train_positive_dataset = ISICDataset(train_positive, phase="train", cfg=cfg)
    elif phase == "valid":
        train_dataset = ISICDataset(train_df, phase="test", cfg=cfg)
        val_dataset = ISICDataset(train_df, phase="test", cfg=cfg)
        test_dataset = ISICDataset(train_df, phase="test", cfg=cfg)
    elif phase == "test":
        train_dataset = ISICDataset(test_df, phase="test", cfg=cfg)
        val_dataset = ISICDataset(test_df, phase="test", cfg=cfg)
        test_dataset = ISICDataset(test_df, phase="test", cfg=cfg)
    elif phase == "vindr":
        df_vindr = ISICDataset.create_dataframe(data_type="vindr")
        train_dataset = ISICDataset(df_vindr, phase="test", cfg=cfg, data_name="vindr")
        val_dataset = ISICDataset(df_vindr, phase="test", cfg=cfg, data_name="vindr")
        test_dataset = ISICDataset(df_vindr, phase="test", cfg=cfg, data_name="vindr")

    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    if phase == "train":
        datasets["train_positive"] = train_positive_dataset
    return datasets
