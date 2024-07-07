from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset


class WrapperDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        transform: Callable,
        phase: str,
        view: Optional[str] = None,
        lat: Optional[str] = None,
    ):
        self.base = base
        self.transform = transform
        self.view = view
        self.lat = lat

    def __len__(self) -> int:
        return len(self.base)

    def apply_transform(self, data):

        image_1 = data.pop("image_1")
        transformed = self.transform(image=image_1)
        image_1 = transformed["image"]  # (1, H, W)

        data["image"] = image_1

        return data

    def __getitem__(self, index: int):
        if self.view is not None or self.lat is not None:
            index = self.idx[index]
        data: dict = self.base[index]
        data = self.apply_transform(data)
        return data
