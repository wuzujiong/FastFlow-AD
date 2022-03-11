import torch
from torch.optim import Adam
from modules.engine import one_epoch, evaluate

def train_model(model, dataloader, test_dataloader, device = torch.device('cuda')):
    """
    For FastFlow, we use and 8-step flows for ResNet18 and Wide-ResNet50-2. We train our model using
    Adam optimizer with the learning rate of 1e-3 and weight decay of 1e-5.
    We use a 500 epoch training schedule, and the batch size is 32.

    eps value is taken from  DifferNet and CS-FLOW
    """
    print(model)
    optimizer = Adam(model.fastflow.parameters(), 1e-3, weight_decay=1e-5)

    epochs = 50
    model.to(device)
    best_auroc = 0.0
    for e in range(epochs):
        train_loss = one_epoch(model, optimizer, dataloader, device)
        auroc, test_loss = evaluate(model, test_dataloader, device)

        if auroc > best_auroc:
            best_auroc = auroc
        print("[Epoch {}/{}]: Training loss: {:.5f} \t Test loss: {:.5f}"
              "\t auroc: {:.5f} \t best auroc: {:.5f} ".format(e, epochs, train_loss,
                                                        test_loss, auroc, best_auroc))