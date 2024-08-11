#!/bin/bash

# GPU番号を引数として受け取る
gpu_index=$1
SCRIPT_NAME=$(basename "$0")

# ハイパーパラメータの組み合わせリスト
declare -a lrs=("1e-5" "2e-5" "5e-5")           # 学習率
declare -a lr_heads=("1e-4" "2e-4")             # ヘッドの学習率
declare -a coarse_dropouts=("0.1" "0.2")        # ドロップアウト率
warmup_ratio="0.1"                              # ウォームアップ比率固定

for lr in "${lrs[@]}"; do
  for lr_head in "${lr_heads[@]}"; do
    for coarse_dropout in "${coarse_dropouts[@]}"; do
      FOLD=0

      # フォルダの名前にパラメータを含める
      out_dir=../results/${SCRIPT_NAME}_lr_${lr}_lrhead_${lr_head}_bs_${batch_size}_ep_${epoch}_cd_${coarse_dropout}_wr_${warmup_ratio}_${FOLD}

      # コマンド実行
      CUDA_VISIBLE_DEVICES=$gpu_index python -m run.train \
        dataset.num_folds=4 \
        dataset.fold_path=./fold/train_with_fold_4.csv \
        dataset.past_fold_path="[./fold/train_2020_with_fold_4.csv,./fold/train_2019_with_fold_4.csv,./fold/train_2018_task_1_2_test_with_fold_4.csv,./fold/train_2018_task_1_2_validation_with_fold_4.csv,./fold/train_2018_task_3_test_with_fold_4.csv,./fold/train_2018_task_3_validation_with_fold_4.csv]" \
        dataset.test_fold=$FOLD \
        dataset.use_only_2024_sites=false \
        dataset.use_cache=false \
        dataset.validation_on_past_data=false \
        dataset.downsampling_rate=0 \
        dataset.downsampling_by_patient=false \
        training.batch_size=32 \
        training.batch_size_test=64 \
        training.epoch=5 \
        training.monitor=val/loss_target \
        preprocessing.h_resize_to=256 \
        preprocessing.w_resize_to=256 \
        preprocessing.mean="[0.485, 0.456, 0.406]" \
        preprocessing.std="[0.229, 0.224, 0.225]" \
        augmentation.use_light_aug=true \
        augmentation.p_coarse_dropout=$coarse_dropout \
        augmentation.coarse_dropout_max_holes=16 \
        augmentation.coarse_dropout_max_size=32 \
        model.base_model=resnet18.fb_swsl_ig1b_ft_in1k \
        training.use_wandb=true \
        training.num_workers=24 \
        forwarder.loss.target_weight=1.0 \
        forwarder.loss.has_lesion_id_weight=0 \
        forwarder.loss.age_scaled_weight=0 \
        forwarder.loss.sex_enc_weight=0 \
        forwarder.loss.anatom_site_general_enc_weight=0 \
        forwarder.loss.tbp_lv_H_weight=0 \
        forwarder.loss.is_past_weight=0 \
        optimizer.lr=$lr \
        optimizer.lr_head=$lr_head \
        scheduler.warmup_steps_ratio=$warmup_ratio \
        training.accumulate_grad_batches=1 \
        out_dir=$out_dir

    done
  done
done
