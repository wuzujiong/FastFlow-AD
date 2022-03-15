import math
from typing import List

from torch import Tensor
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from modules import build_neg_logp, compute_anomaly_scores, build_logp

def one_epoch(model, optimizer, dataloader, device):
    model.fastflow.train()

    train_loss = list()
    for _ in range(4): # sub epochs
        for inputs in dataloader:
            inputs = inputs.to(device)
            with torch.no_grad():
                features = model.feature_extractor(inputs)
            total_loss = .0
            for idx, feat in enumerate(features):
                optimizer.zero_grad()
                z, log_j = model.fastflow[idx](feat)
                logp = build_neg_logp(z, log_j)
                logp.backward()
                optimizer.step()
                total_loss += logp.item()
            train_loss.append(total_loss)

    return np.mean(train_loss)


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.fastflow.eval()

    # detection
    test_labels = list()
    score_labels = list()

    # segmentation
    test_masks = list()
    pred_masks = list()

    for input, mask, y in data_loader:
        input = input.to(device)
        with torch.no_grad():
            features = model.feature_extractor(input)
        distribution: List[Tensor] = []
        for idx, feat in enumerate(features):
            z, log_j = model.fastflow[idx](feat)
            distribution.append(torch.mean(z ** 2 * .5, 1))

        image_score, mask_score = compute_anomaly_scores(distribution, input.shape[-2:])

        test_labels.append(y.item())
        score_labels.append(image_score.item())

    auroc_det = roc_auc_score(test_labels, score_labels)
    auroc_seg = .0
    return auroc_det, auroc_seg
