import os
import shutil
import argparse
from glob import glob

def find_latest_checkpoint_directory(base_dir):
    checkpoint_paths = glob(os.path.join(base_dir, 'isic', '**', 'checkpoints', 'epoch=*-step=*.ckpt'), recursive=True)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {base_dir}")
    
    latest_checkpoint_path = max(checkpoint_paths, key=lambda x: int(x.split('epoch=')[1].split('-')[0]))
    latest_checkpoint_dir = os.path.dirname(latest_checkpoint_path)
    return latest_checkpoint_dir

def main(base_dir):
    for fold in range(5):
        src_dir = f'results/{base_dir}_{fold}'
        dest_dir = os.path.join('sub', f'fold_{fold}')

        # コピー先のディレクトリを作成
        os.makedirs(dest_dir, exist_ok=True)

        # config.yamlのコピー
        src_config = os.path.join(src_dir, '.hydra', 'config.yaml')
        dest_config = os.path.join(dest_dir, 'config.yaml')
        if os.path.exists(src_config):
            print(f"Copying {src_config} to {dest_config}")
            shutil.copy(src_config, dest_config)
        else:
            print(f"Warning: {src_config} does not exist")
        
        # test_result.csvのコピー
        src_result = os.path.join(src_dir, "test_results", "test_results.csv")
        dest_result = os.path.join(dest_dir, "test_results.csv")
        if os.path.exists(src_result):
            print(f"Copying {src_result} to {dest_result}")
            shutil.copy(src_result, dest_result)
        else:
            print(f"Warning: {src_result} does not exist")

        # 最新のチェックポイントディレクトリの確認
        try:
            latest_checkpoint_dir = find_latest_checkpoint_directory(src_dir)
            print(f"Found latest checkpoint directory: {latest_checkpoint_dir}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

        # EMAのモデルウェイトのコピー
        src_ema = os.path.join(latest_checkpoint_dir, 'model_weights_ema.pth')
        dest_ema = os.path.join(dest_dir, 'model_weights_ema.pth')
        if os.path.exists(src_ema):
            print(f"Copying {src_ema} to {dest_ema}")
            shutil.copy(src_ema, dest_ema)
        else:
            print(f"Warning: {src_ema} does not exist in {latest_checkpoint_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize experiment results")
    parser.add_argument('base_dir', type=str, help='Base directory of the experiment')
    args = parser.parse_args()

    main(args.base_dir)
