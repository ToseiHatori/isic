## 使い方
```
pip install -r requirements.txt
```

## opencvでエラーが出たら

```
pip uninstall -y opencv
rm -rf /usr/local/lib/python3.10/dist-packages/cv2/
pip install opencv-python==4.9.0.80
```

## gitでエラーが出たら
```
git config --global --add safe.directory /home/working
```

## 過去のメタデータ収集
```bash
pip install isic-cli
isic metadata download --collections 61 > isic_meta_challenge_2016_test.csv
isic metadata download --collections 74 > isic_meta_challenge_2016_training.csv
isic metadata download --collections 69 > isic_meta_challenge_2017_test.csv
isic metadata download --collections 60 > isic_meta_challenge_2017_training.csv
isic metadata download --collections 71 > isic_meta_challenge_2017_validation.csv
isic metadata download --collections 64 > isic_meta_challenge_2018_task_1-2_test.csv
isic metadata download --collections 63 > isic_meta_challenge_2018_task_1-2_training.csv
isic metadata download --collections 62 > isic_meta_challenge_2018_task_1-2_validation.csv
isic metadata download --collections 67 > isic_meta_challenge_2018_task_3_test.csv
isic metadata download --collections 66 > isic_meta_challenge_2018_task_3_training.csv
isic metadata download --collections 73 > isic_meta_challenge_2018_task_3_validation.csv
isic metadata download --collections 65 > isic_meta_challenge_2019_training.csv
isic metadata download --collections 70 > isic_meta_challenge_2020_training.csv
```
```py
isic_ids2category = {}
for file in sorted(glob.glob("./past_metadata/isic*.csv")):
    df = pd.read_csv(file, low_memory=False)
    category = file.split("isic_meta_")[-1].split(".csv")[0]
    print(f"{category} len {len(df)}")
    for _isic_id in df["isic_id"].values:
        if _isic_id in isic_ids2category:
            # デバッグ用
            #print(_isic_id, category, isic_ids2category[_isic_id])
            isic_ids2category[_isic_id] = category
            pass
        else:
            isic_ids2category[_isic_id] = category
df_past = pd.read_csv("./data/metadata.csv", low_memory=False)
df_past["isic_challenge_category"] = df_past["isic_id"].map(isic_ids2category)
dup = pd.read_csv("./data/2020_Challenge_duplicates.csv")
df_past = df_past.merge(dup[["ISIC_id", "ISIC_id_paired"]], how="left", left_on="isic_id", right_on="ISIC_id")
df_past = df_past[df_past["ISIC_id_paired"].isnull()].reset_index(drop=True)
del df_past["ISIC_id_paired"]
```

bash run/conf/exp/exp_32_resnet18_baseline_5fold_lr.sh 0 
bash run/conf/exp/exp_33_resnet18_baseline_5fold_lr.sh 0 
echo done

bash run/conf/exp/exp_34_efficientnet_b0_baseline_5fold_lr.sh 1 
bash run/conf/exp/exp_35_efficientnet_b0_baseline_5fold_lr.sh 1 

echo done


CUDA_VISIBLE_DEVICES=1 bash run/conf/exp/exp_25_convnext_small_baseline_4_fold_5_epoch_ds_patient_cdp0_lesion_w8.sh 
CUDA_VISIBLE_DEVICES=1 bash run/conf/exp/exp_26_convnext_small_baseline_4_fold_5_epoch_ds_patient_cdp0_lesion_w9.sh 
CUDA_VISIBLE_DEVICES=1 bash run/conf/exp/exp_27_convnext_small_baseline_4_fold_5_epoch_ds_patient_cdp0_lesion_w10.sh 
echo done
