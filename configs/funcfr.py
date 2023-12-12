import torch
import torch.nn as nn

from .registry import ACTIVATION_LAYERS

@ACTIVATION_LAYERS.register_module()
class MyFR(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.act = nn.ReLU6(inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x * self.act(x + 3) / 6), max=8)
