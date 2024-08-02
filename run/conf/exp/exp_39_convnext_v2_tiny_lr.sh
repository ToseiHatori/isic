#!/bin/bash

# GPU番号を引数として受け取る
gpu_index=$1

SCRIPT_NAME=$(basename "$0")

# 学習率のリスト
learning_rates=(1e-5 2e-5 5e-5 1e-4)

for LR in "${learning_rates[@]}"; do
  for FOLD in 0; do
    CUDA_VISIBLE_DEVICES=$gpu_index python -m run.train \
      dataset.num_folds=5 \
      dataset.fold_path=./fold/train_with_fold_5.csv \
      dataset.test_fold=$FOLD \
      dataset.use_cache=false \
      dataset.downsampling_rate=0 \
      dataset.downsampling_by_patient=false \
      training.batch_size=16 \
      training.batch_size_test=32 \
      training.epoch=5 \
      training.monitor=val/pAUC \
      preprocessing.h_resize_to=256 \
      preprocessing.w_resize_to=256 \
      augmentation.use_light_aug=true \
      augmentation.p_coarse_dropout=0.5 \
      augmentation.coarse_dropout_max_holes=16 \
      augmentation.coarse_dropout_max_size=32 \
      model.base_model=convnextv2_tiny.fcmae_ft_in22k_in1k \
      training.use_wandb=true \
      training.num_workers=24 \
      forwarder.loss.has_lesion_id_weight=0.1 \
      optimizer.lr=$LR \
      optimizer.lr_head=2e-4 \
      scheduler.warmup_steps_ratio=0.0 \
      training.accumulate_grad_batches=2 \
      out_dir=../results/${SCRIPT_NAME}_lr_${LR}_${FOLD}
  done
done
