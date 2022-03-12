from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from modules import build_logp

def one_epoch(model, optimizer, dataloader, device):
    model.fastflow.train()

    train_loss = list()
    for inputs in dataloader:
        inputs = inputs.to(device)
        with torch.no_grad():
            features = model.feature_extractor(inputs)
        for idx, feat in enumerate(features):
            optimizer.zero_grad()
            z, j = model.fastflow[idx](feat)
            logp = build_logp(z, j)
            logp.backward()
            optimizer.step()
            train_loss.append(logp.item())

        # optimizer.zero_grad()
        # inputs = inputs.to(device)
        # o1, o2, o3 = model(inputs)
        #
        # loss1 = build_logp(*o1)
        # loss2 = build_logp(*o2)
        # loss3 = build_logp(*o3)
        # loss = (loss1 + loss2 + loss3)
        # loss.backward()
        #
        # # torch.nn.utils.clip_grad_norm_(model.fastflow.parameters(), 1e0)
        # optimizer.step()
        #
        # train_loss.append(loss.item())

    return np.mean(train_loss)

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.fastflow.eval()

    test_loss = list()
    anomaly_score = list()
    test_labels = list()

    for input, mask, y in data_loader:
        input = input.to(device)
        with torch.no_grad():
            features = model.feature_extractor(input)
        score = []
        loss = .0
        for idx, feat in enumerate(features):
            z, j = model.fastflow[idx](feat)
            logp = build_logp(z, j)
            score.append(torch.mean(z.reshape(z.shape[0], -1) ** 2, dim=1).item())
            loss += logp.item()

        anomaly_score.append(score)
        test_loss.append(loss)
        test_labels.append(y.item())

    test_loss = np.mean(test_loss)
    anomaly_score = np.mean(anomaly_score, axis=1)

    return roc_auc_score(test_labels, anomaly_score), test_loss
