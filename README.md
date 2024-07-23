## 使い方
```
pip install -r requirements.txt
bash run/conf/exp/exp_01_convnext_base_baseline_fold_4_fold.sh
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

CUDA_VISIBLE_DEVICES=0 bash run/conf/exp/exp_22_convnext_small_baseline_4_fold_5_epoch_ds_patient_cdp0_lesion_w5.sh 
CUDA_VISIBLE_DEVICES=0 bash run/conf/exp/exp_23_convnext_small_baseline_4_fold_5_epoch_ds_patient_cdp0_lesion_w6.sh 
CUDA_VISIBLE_DEVICES=0 bash run/conf/exp/exp_24_convnext_small_baseline_4_fold_5_epoch_ds_patient_cdp0_lesion_w7.sh 
echo done


CUDA_VISIBLE_DEVICES=1 bash run/conf/exp/exp_25_convnext_small_baseline_4_fold_5_epoch_ds_patient_cdp0_lesion_w8.sh 
CUDA_VISIBLE_DEVICES=1 bash run/conf/exp/exp_26_convnext_small_baseline_4_fold_5_epoch_ds_patient_cdp0_lesion_w9.sh 
CUDA_VISIBLE_DEVICES=1 bash run/conf/exp/exp_27_convnext_small_baseline_4_fold_5_epoch_ds_patient_cdp0_lesion_w10.sh 
echo done
