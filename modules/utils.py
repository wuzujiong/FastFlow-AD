from typing import List, Tuple, Any

import numpy as np
import torch
from torch import Tensor
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torch.nn import functional as F





def compute_anomaly_scores(distribution: List[Tensor], size: tuple) -> Tuple[Any, Any]:
    """
    Calculate the anomaly score and anomaly map.

    Chap3.3: Specifically, we sum the two-dimensional probabilities of each channel to get the final probability map and upsample
    it to the input image resolution using bilinear interpolation.

    Chap4.7: and finally take the average value as the final result.

    :param distribution: distribution of z
    :param img_size: original image size
    :return: anomaly scores
    """

    likelihood_map: List[Tensor] = []
    for likelihood in distribution:
        likelihood -= likelihood.max()  # normalize likelihoods
        likelihood = likelihood.exp()  # convert to probs in range [0:1]
        likelihood_map.append(
            F.interpolate(likelihood.unsqueeze(1), size=size, mode="bilinear", align_corners=False).squeeze()
        )

    # score aggregation
    score_mask = torch.zeros_like(likelihood_map[0])
    for likeli in likelihood_map:
        score_mask += likeli
    score_mask /= 3
    # probs to anomaly scores
    # score_mask = score_mask.max() - score_mask
    score_img = score_mask.max()

    return score_img, score_mask

def get_likelihood(features_dim: int, p_u: torch.Tensor, logdet_j: torch.Tensor) -> torch.Tensor:
    """ Returns the log likelihood estimation """
    ln_sqrt_2pi = -np.log(np.sqrt(2 * np.pi))  # ln(sqrt(2*pi))
    likelihood = features_dim * ln_sqrt_2pi - 0.5 * torch.sum(p_u ** 2, 1) + logdet_j
    likelihood /= features_dim # likelihood per channel
    return likelihood

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








