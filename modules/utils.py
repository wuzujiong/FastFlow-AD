from typing import List, Tuple, Any

import numpy as np
import torch
from torch import Tensor
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torch.nn import functional as F





def compute_anomlay_scores(distribution: List[Tensor], size: tuple) -> Tuple[Any, Any]:
    """
    Calculate the anomaly score and anomaly map.

    Chap3.3: Specifically, we sum the two-dimensional probabilities of each channel to get the final probability map and upsample
    it to the input image resolution using bilinear interpolation.

    Chap4.7: and finally take the average value as the final result.

    :param distribution: distribution of z
    :param img_size: original image size
    :return: anomaly scores
    """

    distribution: List[Tensor] = [
        F.interpolate(d.sum(1).unsqueeze(1), size=size, mode="bilinear", align_corners=True).squeeze()
        for d in distribution
    ]

    score_map = torch.zeros_like(distribution[0])
    for idx in range(len(distribution)):
        score_map += distribution[idx]
    score_map /= len(distribution)

    score_img = score_map.max().item()

    return score_img

def build_logp(z: torch.Tensor, log_j: torch.Tensor) -> torch.Tensor:
    """ Calculate the negative log-likelihood """
    # loss = torch.mean(0.5 * torch.sum(z.reshape(z.shape[0], -1) ** 2, dim=1) - log_j) / z.shape[1]
    loss = torch.mean(0.5 * torch.sum(z ** 2, dim=(1, 2, 3)) - log_j) / z.shape[1]
    return loss

def batch2grid(batch, id_batch, unnmorlaize = True):
    grid = make_grid(batch)
    if unnmorlaize:
        mean = torch.tensor([0.4915, 0.4823, 0.4468])
        std = torch.tensor([0.2470, 0.2435, 0.2616])
        unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        grid = unnormalize(grid)

    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig(f'tests/batch_{id_batch}.png')


class MetricLogger:
    # TODO
    def __init__(self):
        self.train_loss = .0
        self.test_loss = .0
        self.auroc = .0
        self.bestauroc = .0
        self.epoch = .0

    def update(self, train_loss, test_loss, auroc, lr, loss):
       raise NotImplementedError

    def log_score(self, e, epochs, train_loss, test_loss, auroc, best_auroc):
        print("[Epoch {}/{}]: Training loss: {:.5f} \t Test loss: {:.5f}"
              "\t auroc: {:.5f} \t best auroc: {:.5f} ".format(e, epochs, train_loss,
                                                               test_loss, auroc, best_auroc))








