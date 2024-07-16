import pandas as pd
from sklearn.model_selection import GroupKFold

# データの読み込み
train_df = pd.read_csv("./data/train-metadata.csv", low_memory=False)

# Group K-Foldで分割
n_splits = 4
gkf = GroupKFold(n_splits=n_splits)
train_df["fold"] = -1

for fold, (train_idx, val_idx) in enumerate(
    gkf.split(train_df, groups=train_df["patient_id"])
):
    train_df.loc[val_idx, "fold"] = fold

train_df.to_csv("./fold/train_with_fold_4.csv", index=False)
