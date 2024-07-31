#!/bin/bash

# GPU番号を引数として受け取る
gpu_index=$1

SCRIPT_NAME=$(basename "$0")

for FOLD in {0..4}; do
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
    preprocessing.h_resize_to=256 \
    preprocessing.w_resize_to=256 \
    preprocessing.mean="[0.485, 0.456, 0.406]" \
    preprocessing.std="[0.229, 0.224, 0.225]" \
    augmentation.use_light_aug=true \
    model.base_model=efficientnet_b0 \
    training.use_wandb=true \
    training.num_workers=24 \
    forwarder.loss.target_weight=1.0 \
    forwarder.loss.age_scaled_weight=0 \
    forwarder.loss.sex_enc_weight=0 \
    forwarder.loss.anatom_site_general_enc_weight=0 \
    forwarder.loss.has_lesion_id_weight=0 \
    forwarder.loss.tbp_lv_H_weight=0 \
    optimizer.lr=2e-4 \
    optimizer.lr_head=2e-4 \
    scheduler.warmup_steps_ratio=0.0 \
    training.accumulate_grad_batches=2 \
    out_dir=../results/${SCRIPT_NAME}_${FOLD}

done
