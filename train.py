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
    total_samples = 0

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

        # update epoch_loss and total_samples
        batch_size = X.size(0)
        epoch_loss += loss.item() * batch_size
        total_samples += batch_size

    # Calculate the average epoch loss
    epoch_loss /= total_samples

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

    epoch_loss = 0
    total_samples = 0

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

            # update epoch_loss and total_samples
            batch_size = X.size(0)
            epoch_loss += loss.item() * batch_size
            total_samples += batch_size

            # store true and predicted values
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    # Calculate the average epoch loss
    epoch_loss /= total_samples

    # Calculate MSE, RMSE, MAE
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    model.train()

    return epoch_loss, mse, rmse, mae