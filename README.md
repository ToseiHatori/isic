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
python src/utils/clean_past_metadata.py 
python src/utils/get_fold.py --input ./data/past_metadata_challenge_2016_training.csv --output ./fold/train_2016_with_fold_5.csv
python src/utils/get_fold.py --input ./data/past_metadata_challenge_2017_training.csv --output ./fold/train_2017_with_fold_5.csv
python src/utils/get_fold.py --input ./data/past_metadata_challenge_2018_task_1-2_training.csv --output ./fold/train_2018_task_1_2_train_with_fold_5.csv
python src/utils/get_fold.py --input ./data/past_metadata_challenge_2018_task_1-2_validation.csv --output ./fold/train_2018_task_1_2_validation_with_fold_5.csv
python src/utils/get_fold.py --input ./data/past_metadata_challenge_2018_task_1-2_test.csv --output ./fold/train_2018_task_1_2_test_with_fold_5.csv
python src/utils/get_fold.py --input ./data/past_metadata_challenge_2018_task_3_training.csv --output ./fold/train_2018_task_3_train_with_fold_5.csv
python src/utils/get_fold.py --input ./data/past_metadata_challenge_2018_task_3_validation.csv --output ./fold/train_2018_task_3_validation_with_fold_5.csv
python src/utils/get_fold.py --input ./data/past_metadata_challenge_2018_task_3_test.csv --output ./fold/train_2018_task_3_test_with_fold_5.csv
python src/utils/get_fold.py --input ./data/past_metadata_challenge_2019_training.csv --output ./fold/train_2019_with_fold_5.csv
python src/utils/get_fold.py --input ./data/past_metadata_challenge_2020_training.csv --output ./fold/train_2020_with_fold_5.csv


```
