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