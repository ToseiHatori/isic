# 5th Place Winning Solution - ISIC 2024 - Skin Cancer Detection with 3D-TBP
This is the repository of part of 5th winning solution of "ISIC 2024 - Skin Cancer Detection with 3D-TBP" in kaggle.
https://www.kaggle.com/competitions/isic-2024-challenge/discussion/533056

![solution](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3948967%2F38ce6eb11d8320552850efd2a20a1d87%2Fimage1.png?generation=1725877390031631&alt=media)

## Download past competition data
- https://www.kaggle.com/datasets/tomooinubushi/all-isic-data-20240629
- download past_metadata.csv and image_256sq.hdf5 on `data` folder
## Environments
- host
```bash
docker compose up -d
docker exec -it isic bash
```
- in docker container
```bash
pip install -r requirements.txt
# nvidia docker opencv problem...
pip uninstall -y opencv
rm -rf /usr/local/lib/python3.10/dist-packages/cv2/
pip install opencv-python==4.9.0.80
# git setting
git config --global --add safe.directory /home/working
```

## Prepare data
- in docker container
```py
python src/utils/get_fold.py data/train-metadata.csv fold/train_with_fold_5.csv 5
python src/utils/get_fold.py data/past_metadata.csv fold/train_past_metadata_with_fold_5.csv 5
```

## Training
- in docker container
```bash
bash run/conf/exp/exp_77_resnet152_baseline_5fold_ex_all_validation_on_past_all_w_1e-4.sh 0
```
- We finally used exp71, 73, 75, 77.
- Trained models
  - https://www.kaggle.com/datasets/toseihatori/isic-2024-sub-models/data

## Hardware
- **Operating System**: Ubuntu 22.04.4 LTS
- **Graphics**: Nvidia RTX3090

## References
- https://github.com/tyamaguchi17/rsna_mammo
- https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175412