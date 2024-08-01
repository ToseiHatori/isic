import logging

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# データの読み込み
train_df = pd.read_csv("./data/train-metadata.csv", low_memory=False)

# Stratified Group K-Foldで分割
n_splits = 5
sgkf = StratifiedGroupKFold(n_splits=n_splits)
train_df["fold"] = -1

for fold, (train_idx, val_idx) in enumerate(
    sgkf.split(train_df, train_df["target"], groups=train_df["patient_id"])
):
    train_df.loc[val_idx, "fold"] = fold

    # foldごとの件数や目的変数の件数をロギング
    fold_size = len(val_idx)
    target_counts = train_df.loc[val_idx, "target"].value_counts().to_dict()

    logger.info(f"Fold {fold}")
    logger.info(f"Total samples in fold: {fold_size}")
    logger.info(f"Target counts in fold: {target_counts}")

# fold = -1 が残っていないか確認
assert -1 not in train_df["fold"].values, "There are samples with fold = -1"

# 結果を保存
train_df.to_csv(f"./fold/train_with_fold_{n_splits}.csv", index=False)
