for FOLD in {0..3}; do
python -m run.train \
  dataset.num_folds=4 \
  dataset.test_fold=$FOLD \
  dataset.use_cache=false \
  dataset.fold_path=./fold/train_with_fold_4.csv \
  training.batch_size=16 \
  training.batch_size_test=64 \
  training.epoch=20 \
  preprocessing.h_resize_to=256 \
  preprocessing.w_resize_to=256 \
  augmentation.use_light_aug=true \
  model.base_model=convnext_base.fb_in22k_ft_in1k \
  training.use_wandb=true \
  training.num_workers=24 \
  optimizer.lr=5e-5 \
  optimizer.lr_head=5e-4 \
  scheduler.warmup_steps_ratio=0.0 \
  training.accumulate_grad_batches=2 \
  out_dir=../results/01_convnext_base_baseline_fold_$FOLD; done 
