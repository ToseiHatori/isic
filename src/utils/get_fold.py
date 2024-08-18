import argparse
import logging

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# コマンドライン引数の設定
parser = argparse.ArgumentParser(
    description="Stratified Group K-Fold split for ISIC dataset."
)
parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
parser.add_argument(
    "--n_splits", type=int, default=5, help="Number of splits for K-Fold"
)
args = parser.parse_args()

# 引数の取得
input_filename = args.input
output_filename = args.output
n_splits = args.n_splits

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# データの読み込み
train_df = pd.read_csv(input_filename, low_memory=False)

if "target" not in train_df.columns:
    # 過去データ用の処理
    train_df = train_df[~train_df['benign_malignant'].isnull()].reset_index(drop=True)
    target_map = {'benign': 0, 'indeterminate': 0, 'indeterminate/benign': 0,
              'indeterminate/malignant': 1, 'malignant': 1}
    train_df['target'] = train_df['benign_malignant'].map(target_map)
    # patient_idがnullであることがしばしばあるのでisic_idで埋めておく
    idx = train_df["patient_id"].isnull()
    train_df.loc[idx, "patient_id"] = train_df.loc[idx, "isic_id"]

# Stratified Group K-Foldで分割
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
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
train_df.to_csv(output_filename, index=False)
