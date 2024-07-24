FOLD=0
SCRIPT_NAME=$(basename "$0")

gpu_index=0

for has_lesion_id_weight in $(seq 0.4 0.1 1.1); do
  for p_coarse_dropout in 0.5; do
    for coarse_dropout_max_holes in 16; do
      for coarse_dropout_max_size in 32; do
        for downsampling_by_patient in false; do
          # パラメータを10倍して整数に変換
          has_lesion_id_weight_int=$(awk -v value="$has_lesion_id_weight" 'BEGIN { printf "%d\n", value * 10 }')
          p_coarse_dropout_int=$(awk -v value="$p_coarse_dropout" 'BEGIN { printf "%d\n", value * 10 }')
          
          CUDA_VISIBLE_DEVICES=$gpu_index python -m run.train \
            dataset.num_folds=4 \
            dataset.test_fold=$FOLD \
            dataset.use_cache=false \
            dataset.downsampling_rate=0 \
            dataset.downsampling_by_patient=$downsampling_by_patient \
            training.batch_size=16 \
            training.batch_size_test=32 \
            training.epoch=5 \
            preprocessing.h_resize_to=256 \
            preprocessing.w_resize_to=256 \
            augmentation.use_light_aug=true \
            augmentation.p_coarse_dropout=$p_coarse_dropout \
            augmentation.coarse_dropout_max_holes=$coarse_dropout_max_holes \
            augmentation.coarse_dropout_max_size=$coarse_dropout_max_size \
            model.base_model=convnext_small.fb_in22k_ft_in1k \
            training.use_wandb=true \
            training.num_workers=24 \
            forwarder.loss.has_lesion_id_weight=$has_lesion_id_weight \
            optimizer.lr=2e-5 \
            optimizer.lr_head=2e-4 \
            scheduler.warmup_steps_ratio=0.0 \
            training.accumulate_grad_batches=2 \
            out_dir=../results/${SCRIPT_NAME}_${FOLD}_w${has_lesion_id_weight_int}_p${p_coarse_dropout_int}_h${coarse_dropout_max_holes}_s${coarse_dropout_max_size}_d${downsampling_by_patient} &

          gpu_index=$((1 - gpu_index)) # 0と1を交互に切り替え

          # 並列実行のためにバックグラウンドジョブが2つ以上ある場合は待機
          if [[ $(jobs -r -p | wc -l) -ge 2 ]]; then
            wait -n
          fi
        done
      done
    done
  done
done

# 最後にすべてのジョブが完了するのを待つ
wait
