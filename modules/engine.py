from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from modules import build_logp

def one_epoch(model, optimizer, dataloader, device):
    model.fastflow.train()

    train_loss = list()
    for inputs in dataloader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        o1, o2, o3 = model(inputs)

        loss1 = build_logp(*o1)
        loss2 = build_logp(*o2)
        loss3 = build_logp(*o3)
        loss = (loss1 + loss2 + loss3)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.fastflow.parameters(), 1e0)
        optimizer.step()

        train_loss.append(loss.item())

    return np.mean(train_loss)

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.fastflow.eval()

    test_loss = list()
    anomaly_score = list()
    test_labels = list()

    for input, mask, y in data_loader:
        input = input.to(device)
        o1, o2, o3 = model(input)

        # Compute loss score
        loss1 = build_logp(*o1)
        loss2 = build_logp(*o2)
        loss3 = build_logp(*o3)
        test_loss.append(loss1.item() + loss2.item() + loss3.item())

        # Compute anomaly score

        z1 = o1[0].reshape(o1[0].shape[0], -1)
        z2 = o2[0].reshape(o2[0].shape[0], -1)
        z3 = o3[0].reshape(o3[0].shape[0], -1)

        score1 = torch.mean(z1 ** 2, dim = 1)
        score2 = torch.mean(z2 ** 2, dim = 1)
        score3 = torch.mean(z3 ** 2, dim = 1)

        anomaly_score.append([score1.item(), score2.item(), score3.item()])

        test_labels.append(y.item())

    test_loss = np.mean(test_loss)
    anomaly_score = np.mean(anomaly_score, axis=1)

    return roc_auc_score(test_labels, anomaly_score), test_loss
