import random
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import pandas as pd
from pfio.cache import MultiprocessFileCache
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

ANATOM_SITE_GENERAL_ENCODER = {
    "unk": 0,
    "anterior torso": 1,
    "head/neck": 2,
    "lower extremity": 3,
    "posterior torso": 4,
    "upper extremity": 5,
}
SEX_ENCODER = {"male": 0, "female": 1}


class ISICDataset(Dataset):

    ROOT_PATH = Path("./data")

    @classmethod
    def create_dataframe(
        cls,
        num_folds: int = 4,
        seed: int = 2023,
        num_records: int = 0,
        fold_path: Optional[str] = "./fold/train_with_fold.csv",
        data_type: str = "train",
        pl_path: Optional[str] = None,
    ) -> pd.DataFrame:
        root = cls.ROOT_PATH

        if data_type == "train":
            if fold_path is not None:
                df = pd.read_csv(fold_path, low_memory=False)
                if num_records:
                    df = df[:num_records]
                return df
            else:
                # not supported
                df = pd.read_csv(str(root / "train.csv"))
                assert 0 == 1

        elif data_type == "test":
            # not supported
            df = pd.read_csv(str(root / "sample_submission.csv"))
            assert 0 == 1
            return df

        n_splits = num_folds
        shuffle = True

        kfold = GroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        X = df["isic_id"].values
        y = df["cancer"].values
        group = df["patient_id"].values
        fold = -np.ones(len(df))
        for i, (_, indices) in enumerate(kfold.split(X, y, group=group)):
            fold[indices] = i

        df["fold"] = fold

        if num_records:
            df = df[::num_records]

        return df

    def __init__(
        self,
        df: pd.DataFrame,
        phase="train",
        cfg=None,
        data_name="isic",
    ) -> None:
        self.df = df.copy()
        self.df["original_index"] = df.index
        self.df.reset_index(inplace=True)
        # 前処理系(ここでやるべきではないとは思っている)
        self.df["age_scaled"] = (
            self.df["age_approx"].fillna(60) / 90
        )  # 後でloglossで評価するのでクソでかい数字で割っておく
        self.df["sex"] = self.df["sex"].fillna("male")
        self.df["anatom_site_general"] = self.df["anatom_site_general"].fillna("unk")
        self.df["has_lesion_id"] = self.df["lesion_id"].notnull().astype(int)
        # 最小値が-1.5くらい、最大値が105くらいなので[0, 1]に収まるように
        self.df["tbp_lv_H"] = (self.df["tbp_lv_H"] + 2) / 110

        # 0fill
        self.df = self.df.infer_objects()  # これやっとかないと警告が出る
        self.df = self.df.fillna(0)
        self.df["anatom_site_general_enc"] = self.df["anatom_site_general"].map(
            ANATOM_SITE_GENERAL_ENCODER
        )
        self.df["sex_enc"] = self.df["sex"].map(SEX_ENCODER)
        # IDがstringだとGPUに乗らなくてDDPできないのでintにしておく
        # self.df["isic_id_int"] = self.df["isic_id"].map(lambda x: int(x.split("ISIC_")[1]))
        # self.df["patient_id_int"] = self.df["patient_id"].map(lambda x: int(x.split("IP_")[1]))
        self.data_name = data_name
        self.root = self.ROOT_PATH
        self.phase = phase
        self.cfg_aug = cfg.augmentation
        self.fp_hdf = h5py.File("./data/train-image.hdf5", mode="r")

        if cfg.use_cache:
            cache_dir = "/tmp/isic/"
            self._cache = MultiprocessFileCache(
                len(self), dir=cache_dir, do_pickle=True
            )
        else:
            cache_dir = None
            self._cache = None

    def __len__(self) -> int:
        return len(self.df)

    def _read_image(self, index):
        root = self.ROOT_PATH
        image_id = self.df.at[index, "isic_id"]
        if self.data_name == "isic" or self.data_name == "vindr":
            # 画像データを読み込む（例: self.fp_hdf[image_id][()]）
            image_data = self.fp_hdf[image_id][()]
            # BytesIOオブジェクトを作成
            image_stream = BytesIO(image_data)
            # BytesIOからバイトデータを取得
            byte_array = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
            # OpenCVで画像データをデコード
            image = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)
            return image, 0

    def read_image(self, index):
        if self._cache:
            image, _ = self._cache.get_and_cache(index, self._read_image)
        else:
            image, _ = self._read_image(index)
        return image

    def get_metadata(self, index):
        res = {
            "age_scaled": self.df.loc[index, "age_scaled"],
            "sex_enc": self.df.loc[index, "sex_enc"],
            "anatom_site_general_enc": self.df.loc[index, "anatom_site_general_enc"],
            "has_lesion_id": self.df.loc[index, "has_lesion_id"],
            "tbp_lv_H": self.df.loc[index, "tbp_lv_H"],
        }
        return res

    def __getitem__(self, index: int):

        label = self.df.loc[index, "target"]
        isic_id = self.df.loc[index, "isic_id"]
        patient_id = self.df.loc[index, "patient_id"]

        image_1 = self.read_image(index)
        metadata = self.get_metadata(index)
        res = {
            "original_index": self.df.at[index, "original_index"],
            "isic_id": isic_id,
            "patient_id": patient_id,
            "label": label,
            "image_1": image_1,
        }
        res.update(metadata)

        return res
