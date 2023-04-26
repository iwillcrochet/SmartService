import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_fn(loader, model, loss_fn, device, optimizer, scheduler):
    """Train function for training the model

    :param loader   (torch.utils.data.DataLoader): training dataloader
    :param model    (torch.nn.Module): model to train
    :param optimizer(torch.optim): optimizer to use

    :param loss_fn  (torch.nn.Module): loss function to use
    :param device   (torch.device): device to train on (CPU or GPU)
    :param epoch    (int): current epoch number

    :return: None
    """

    # set model to train mode
    model.train()

    epoch_loss = 0
    num_batches = 0

    # iterate over batches
    for batch_idx, (X, y) in enumerate(loader):
        # put data and targets to device
        X = X.to(device)
        y = y.to(device)

        # forward pass
        preds = model(X)
        # calculate loss
        loss = loss_fn(preds, y)

        # zero out gradients
        optimizer.zero_grad()

        # backpropagate
        loss.backward()

        # update optimizer
        optimizer.step()

        # step scheduer on batch
        scheduler.step()

        # update epoch_loss + observation count
        epoch_loss += loss.item()
        num_batches += 1

    # calculate average epoch loss
    epoch_loss = epoch_loss / num_batches

    return epoch_loss


def eval_fn(loader, model, loss_fn, device):
    """Evaluation function for evaluating the model

    :param loader   (torch.utils.data.DataLoader): testing dataloader
    :param model    (torch.nn.Module): model to evaluate
    :param device   (torch.device): device to evaluate on (CPU or GPU)
    :param loss_fn  (torch.nn.Module): loss function to use

    :return: mean squared error (float), root mean squared error (float), mean absolute error (float), test_epoch_loss (float)
    """

    # set model to evaluation mode
    model.eval()

    y_true = []
    y_pred = []

    test_epoch_loss = 0
    num_batches = 0

    # disable gradient calculation
    with torch.inference_mode():
        for batch_idx, (X, y) in enumerate(loader):
            # put data and targets to device
            X = X.to(device)
            y = y.to(device)

            # forward pass
            preds = model(X)

            # calculate loss
            loss = loss_fn(preds, y)
            test_epoch_loss += loss.item()

            # store true and predicted values
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

            # update batch count
            num_batches += 1

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # calculate average test loss
    test_epoch_loss = test_epoch_loss / num_batches

    model.train()

    return test_epoch_loss, mse, rmse, mae