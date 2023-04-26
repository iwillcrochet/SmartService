import torch
import random
import numpy as np
import pandas as pd

def main():
    ############################
    # Hyperparameters
    ############################
    RANDOM_SEED = 90
    LEARNING_RATE = 1e-4 # (0.0001)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    NUM_EPOCHS = 200
    WARMUP_EPOCHS = int(NUM_EPOCHS * 0.05) # 5% of the total epochs
    if DEVICE == "cuda":
        NUM_WORKERS = 4
    else:
        NUM_WORKERS = 0
    PIN_MEMORY = True

    ############################
    # set seeds
    ############################
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = False

    ############################
    # load data frame and declare features and target
    ############################
    # load data frame
    df = pd.read_csv('diabetes.csv')
    print(df.head())

    # target
    y = df['BMI']

    # feature -> all columns except the target
    X = df.drop('BMI', axis=1)

    # split into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # specify input and output sizes
    INPUT_SIZE = X_train.shape[1]
    OUTPUT_SIZE = 1

    ############################
    # create data loaders
    ############################

    # import data loader
    from torch.utils.data import DataLoader

    # create training data loader
    train_data_loader = DataLoader(
        dataset=X_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # create testing data loader
    test_data_loader = DataLoader(
        dataset=X_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    ############################
    # create model
    ############################
    # instantiate model
    from model import FullyConnectedModel
    NUM_HIDDEN_LAYERS = 2
    NODES_PER_LAYER = 10

    model = FullyConnectedModel(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        nodes_per_layer=NODES_PER_LAYER
    )
    model.to(DEVICE)

    # loss function -> RMSE
    from torch import nn
    loss_fn = nn.MSELoss(reduction='mean')

    # optimizer -> Adam
    from torch import optim
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # scheduler -> cosine annealing with warm restarts
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=WARMUP_EPOCHS,
        T_mult=1,
        eta_min=LEARNING_RATE * 1e-3
    )

    ############################
    # train model
    ############################
    # import train function
    from train import train_fn, eval_fn

    # Train and evaluate the model
    for epoch in range(NUM_EPOCHS):
        train_loss = train_fn(train_data_loader, model, optimizer, loss_fn, DEVICE, epoch, scheduler)

        test_loss, mse, rmse, mae = eval_fn(test_data_loader, model, DEVICE)

        # print training loss and test metrics:
        print(
            f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Test Loss:{test_loss:.4f}, Test MSE: {mse:.2f}, Test RMSE: {rmse:.2f}, Test MAE: {mae:.2f}")

    ############################
    # save model
    ############################
    # save model
    torch.save(model.state_dict(), 'model.pth')
