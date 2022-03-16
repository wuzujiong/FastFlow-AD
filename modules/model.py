from collections import OrderedDict
from typing import List, Callable
import timm
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
        inn.append(Fm.AllInOneBlock, reverse_permutation = True, subnet_constructor=subnet_conv_3x3)
        inn.append(Fm.AllInOneBlock, reverse_permutation = True, subnet_constructor=subnet_conv_1x1)

    return inn


class FastFlow(nn.Module):
    def __init__(self, backbone = 'wide_resnet50_2'):
        """
        For resnet we directly use the features of the last layer in the first three blocks
        and put these features into three corresponding FastFlow model

        :param backbone: wide_resnet50_2 and resnet18 are supported
        """

        assert backbone == 'wide_resnet50_2' or backbone == 'resnet18'

        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=True, features_only = True, out_indices=(1, 2, 3))
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Dry run to initialize the flow heads
        with torch.no_grad():
            features = self.encoder(torch.randn(1, 3, 256, 256))
        # discussion: https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/4
        self.fastflow = nn.ModuleList([
            fastflow_head(feat.shape[1:]) for feat in features
        ])

    def forward(self, x: List[Tensor]) -> List:
        with torch.no_grad():
            features = self.encoder(x)
        # features = list(features.values())

        return [head_flow(feat) for feat, head_flow in zip (features, self.fastflow)]

