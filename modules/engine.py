import math
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from modules import build_logp, compute_anomaly_scores, get_likelihood

def one_epoch(model, optimizer, dataloader, device):
    model.fastflow.train()

    train_loss = list()
    for _ in range(4): # sub epochs
        for inputs in dataloader:
            inputs = inputs.to(device)
            with torch.no_grad():
                features = model.feature_extractor(inputs)
            total_loss = torch.zeros([1], device=device)
            for idx, feat in enumerate(features):
                optimizer.zero_grad()
                z, log_j = model.fastflow[idx](feat)
                logp = build_logp(z, log_j)
                logp.backward()
                optimizer.step()
                total_loss += logp.detach()
            train_loss.append(total_loss.item())

    return np.mean(train_loss)


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.fastflow.eval()

    # detection
    test_labels = list()
    score_labels = list()

    for input, mask, label in data_loader:
        input = input.to(device)
        with torch.no_grad():
            features = model.feature_extractor(input)
        scores = torch.zeros(len(features), device=device)
        for idx, feat in enumerate(features):
            z, log_j = model.fastflow[idx](feat)
            z = z.reshape(z.shape[0], -1)
            scores[idx] = torch.mean(z ** 2 / 2, 1)

        score_labels.append(scores.mean().item())
        test_labels.append(label.item())


    auroc_det = roc_auc_score(test_labels, score_labels)
    auroc_seg = .0
    return auroc_det, auroc_seg
