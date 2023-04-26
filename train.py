import tqdm
import torch

def train_fn(loader, model, optimizer, loss_fn, device, epoch, scheduler):
    """Train function for training the model

    :param loader   (torch.utils.data.DataLoader): training dataloader
    :param model    (torch.nn.Module): model to train
    :param optimizer(torch.optim): optimizer to use

    :param loss_fn  (torch.nn.Module): loss function to use
    :param device   (torch.device): device to train on (CPU or GPU)
    :param epoch    (int): current epoch number

    :return: None
    """

    # set tqdm loop with epoch number
    loop = tqdm(loader, desc=f"Epoch {epoch}")

    # set model to train mode
    model.train()

    epoch_loss = 0
    num_batches = len(loader)

    # iterate over batches
    for batch_idx, (X, y) in enumerate(loop):
        # convert data and targets to tensors and move them to the device
        X = torch.tensor(X.values, dtype=torch.float32).to(device)
        y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)

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

        epoch_loss += loss.item()

        # update tqdm loop
        loop.set_postfix(loss=f"{loss.item():.4f}")

        # step scheduer on batch
        scheduler.step()

    # calculate average epoch loss
    epoch_loss = epoch_loss / num_batches

    # update tqdm loop
    loop.set_postfix(loss=f"{epoch_loss:.4f}")

    return epoch_loss

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def eval_fn(loader, model, device, loss_fn):
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
    num_batches = len(loader)

    # disable gradient calculation
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loader):
            # convert data and targets to tensors and move them to the device
            X = torch.tensor(X.values, dtype=torch.float32).to(device)
            y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(device)

            # forward pass
            preds = model(X)

            # calculate loss
            loss = loss_fn(preds, y)
            test_epoch_loss += loss.item()

            # store true and predicted values
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # calculate average test loss
    test_epoch_loss = test_epoch_loss / num_batches

    return test_epoch_loss, mse, rmse, mae




