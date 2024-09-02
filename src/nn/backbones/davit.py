from timm.models.davit import DaVit
from torch import Tensor

from .base import BackboneBase


class DaVitBackbone(BackboneBase):
    def __init__(self, model: DaVit) -> None:
        assert isinstance(model, DaVit)
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        x = self.model.forward_features(x)
        return x

    @property
    def out_features(self) -> int:
        return self.model.num_features
