#!/bin/bash

# GPU番号を引数として受け取る
gpu_index=$1
SCRIPT_NAME=$(basename "$0")

for FOLD in {0..4}; do
  CUDA_VISIBLE_DEVICES=$gpu_index python -m run.train \
    dataset.num_folds=5 \
    dataset.fold_path=./fold/train_with_fold_5.csv \
    dataset.past_fold_path="[./fold/train_past_metadata_with_fold_5.csv]" \
    dataset.test_fold=$FOLD \
    dataset.use_only_2024_sites=false \
    dataset.use_cache=false \
    dataset.validation_on_past_data=true \
    dataset.downsampling_rate=0 \
    dataset.downsampling_by_patient=false \
    training.batch_size=32 \
    training.batch_size_test=64 \
    training.epoch=5 \
    training.monitor=val/pAUC \
    preprocessing.h_resize_to=256 \
    preprocessing.w_resize_to=256 \
    preprocessing.mean="[0.485, 0.456, 0.406]" \
    preprocessing.std="[0.229, 0.224, 0.225]" \
    augmentation.use_light_aug=true \
    augmentation.p_coarse_dropout=0.1 \
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
    optimizer.lr=5e-5 \
    optimizer.lr_head=1e-4 \
    scheduler.warmup_steps_ratio=0.1 \
    training.accumulate_grad_batches=1 \
    out_dir=../results/${SCRIPT_NAME}_${FOLD}
done
