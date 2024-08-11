#!/bin/bash

# GPU番号を引数として受け取る
gpu_index=$1
SCRIPT_NAME=$(basename "$0")

for FOLD in {0..3}; do
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
    training.batch_size=16 \
    training.batch_size_test=32 \
    training.epoch=5 \
    training.monitor=val/pAUC \
    preprocessing.h_resize_to=256 \
    preprocessing.w_resize_to=256 \
    preprocessing.mean="[0.485, 0.456, 0.406]" \
    preprocessing.std="[0.229, 0.224, 0.225]" \
    augmentation.use_light_aug=true \
    augmentation.p_coarse_dropout=0.2 \
    augmentation.coarse_dropout_max_holes=16 \
    augmentation.coarse_dropout_max_size=32 \
    model.base_model=convnext_small.fb_in22k_ft_in1k \
    training.use_wandb=true \
    training.num_workers=24 \
    forwarder.loss.target_weight=1.0 \
    forwarder.loss.has_lesion_id_weight=0.1 \
    forwarder.loss.age_scaled_weight=0 \
    forwarder.loss.sex_enc_weight=0 \
    forwarder.loss.anatom_site_general_enc_weight=0 \
    forwarder.loss.tbp_lv_H_weight=0 \
    forwarder.loss.is_past_weight=1e-4 \
    optimizer.lr=2e-5 \
    optimizer.lr_head=2e-4 \
    scheduler.warmup_steps_ratio=0.0 \
    training.accumulate_grad_batches=2 \
    out_dir=../results/${SCRIPT_NAME}_${FOLD}
done
