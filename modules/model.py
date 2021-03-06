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

def nf_head(dims: tuple) -> Ff.SequenceINN:

    inn = Ff.SequenceINN(*dims)
    for k in range(4):
        inn.append(Fm.AllInOneBlock, reverse_permutation = True, subnet_constructor=subnet_conv_3x3)
        inn.append(Fm.AllInOneBlock, reverse_permutation = True, subnet_constructor=subnet_conv_1x1)
    return inn

class FastFlow(nn.Module):
    def __init__(self):
        """
        For resnet we directly use the features of the last layer in the first three blocks
        and put these features into three corresponding FastFlow model
        """

        super().__init__()
        backbone = models.wide_resnet50_2(True, True)
        self.encoder = create_feature_extractor(backbone, {f'layer{k}': str(i) for i,k in enumerate([1, 2, 3])})
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Dry run to initialize the flow heads
        with torch.no_grad():
            features = self.encoder(torch.randn(1, 3, 256, 256))

        # discussion: https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/4
        self.decoder = nn.ModuleList([
            nf_head(features[layer].shape[1:]) for layer in features
        ])

    def forward(self, x: List[Tensor]):
        raise NotImplementedError
