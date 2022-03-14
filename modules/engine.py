from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from modules import build_logp, compute_anomlay_scores

def one_epoch(model, optimizer, dataloader, device):
    model.fastflow.train()

    train_loss = list()
    for _ in range(4): # sub epochs
        for inputs in dataloader:
            inputs = inputs.to(device)
            with torch.no_grad():
                features = model.feature_extractor(inputs)
            loss = torch.zeros([1], device=device)
            for idx, feat in enumerate(features):
                optimizer.zero_grad()
                z, j = model.fastflow[idx](feat)
                logp = build_logp(z, j)
                logp.backward()
                optimizer.step()
                loss += logp.detach()
            train_loss.append(loss.item())

    return np.mean(train_loss)

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.fastflow.eval()

    test_loss = list()
    test_labels = list()
    score_labels = list()
    score_mask = list() # currently, it didn't use

    for input, mask, y in data_loader:
        input = input.to(device)
        with torch.no_grad():
            features = model.feature_extractor(input)
        distribution = []
        loss = torch.zeros([1], device=device)
        for idx, feat in enumerate(features):
            z, j = model.fastflow[idx](feat)
            logp = build_logp(z, j)
            loss += logp.detach()
            distribution.append(z)

        score_img = compute_anomlay_scores(distribution, size=(256, 256))
        score_labels.append(score_img)
        test_loss.append(loss.item())
        test_labels.append(y.item())

    auroc = roc_auc_score(test_labels, score_labels)
    return np.mean(test_loss), auroc
