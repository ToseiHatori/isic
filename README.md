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
