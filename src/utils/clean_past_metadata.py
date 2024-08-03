import glob

import pandas as pd

isic_ids2category = {}
for file in sorted(glob.glob("./past_metadata/isic*.csv")):
    df = pd.read_csv(file, low_memory=False)
    category = file.split("isic_meta_")[-1].split(".csv")[0]
    print(f"{category} len {len(df)}")
    for _isic_id in df["isic_id"].values:
        if _isic_id in isic_ids2category:
            # デバッグ用
            # print(_isic_id, category, isic_ids2category[_isic_id])
            isic_ids2category[_isic_id] = category
            pass
        else:
            isic_ids2category[_isic_id] = category

df_past = pd.read_csv("./data/metadata.csv", low_memory=False)
print(df_past.shape)
df_past["isic_challenge_category"] = df_past["isic_id"].map(isic_ids2category)
dup = pd.read_csv("./data/2020_Challenge_duplicates.csv")
df_past = df_past.merge(
    dup[["ISIC_id", "ISIC_id_paired"]],
    how="left",
    left_on="isic_id",
    right_on="ISIC_id",
)
df_past = df_past[df_past["ISIC_id_paired"].isnull()]
df_past = df_past[df_past["benign_malignant"].notnull()].reset_index(drop=True)
del df_past["ISIC_id_paired"], df_past["ISIC_id"]

# patient_idが空なことがあるので代わりにisic_idを入れておく
idx = df_past["patient_id"].isnull()
df_past.loc[idx, "patient_id"] = df_past.loc[idx, "isic_id"]
# targetを作っておく
df_past["target"] = df_past["benign_malignant"].map(
    {
        "benign": 0,
        "malignant": 1,
        "indeterminate": 0,
        "indeterminate/malignant": 1,
        "indeterminate/benign": 0,
    }
)

print(df_past.shape)
for cat in df_past["isic_challenge_category"].unique():
    print(cat)
    df_past_cat = df_past[df_past["isic_challenge_category"] == cat].reset_index(
        drop=True
    )
    df_past.to_csv(f"./data/past_metadata_{cat}.csv")
