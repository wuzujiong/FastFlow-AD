import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision import transforms as T

def build_logp(z: torch.Tensor, log_j: torch.Tensor) -> torch.Tensor:
    """ Calculate the negative log-likelihood """
    z = z.reshape(z.shape[0], -1)
    loss = torch.mean(0.5 * torch.sum(z ** 2, dim=1) - log_j) / z.shape[1]
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








