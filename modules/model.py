from typing import List, Callable

import torch
from torch import nn, Tensor

from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

import FrEIA.modules as Fm
import FrEIA.framework as Ff


def subnet_conv_3x3(channels_in: int, channels_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels_in, 2*channels_in, (3, 3), padding=1),
        nn.ReLU(),
        nn.Conv2d(2*channels_in, channels_out, (3, 3), padding=1)
    )

def subnet_conv_1x1(channels_in: int, channels_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels_in, 2*channels_in, (1, 1)),
        nn.ReLU(),
        nn.Conv2d(2*channels_in, channels_out, (1, 1))
    )


def fastflow_head(dims: tuple) -> Ff.SequenceINN:

    inn = Ff.SequenceINN(*dims)
    for k in range(4):
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_conv_3x3, affine_clamping = 2.5)
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_conv_1x1, affine_clamping = 2.5)

    return inn


class FastFlow(nn.Module):
    def __init__(self):
        """
        For resnet we directly use the features of the last layer in the first three blocks
        and put these features into three corresponding FastFlow model
        """

        super().__init__()
        backbone = models.wide_resnet50_2(True)
        return_nodes = ['layer1', 'layer2', 'layer3']
        self.feature_extractor = create_feature_extractor(backbone,
                                                          return_nodes=return_nodes)
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # discussion: https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/4
        self.fastflow = nn.ModuleList([
            fastflow_head(dim)
            for dim in ((256, 64, 64), (512, 32, 32), (1024, 16, 16)) # hardcoded at the moment
        ])

    def forward(self, x: List[Tensor]) -> List:
        with torch.no_grad():
            x = self.feature_extractor(x)
        return [head_flow(x[key]) for key, head_flow in zip (x, self.fastflow)]

