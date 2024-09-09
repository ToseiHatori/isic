from timm.models.maxxvit import MaxxVit
from torch import Tensor

from .base import BackboneBase


class MaxxVitBackbone(BackboneBase):
    def __init__(self, model: MaxxVit) -> None:
        assert isinstance(model, MaxxVit)
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        x = self.model.forward_features(x)
        return x

    @property
    def out_features(self) -> int:
        return self.model.num_features
