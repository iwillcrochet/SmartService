import torch
import random
import numpy as np
import pandas as pd

def main():
    ############################
    # Hyperparameters
    ############################
    RANDOM_SEED = 42
    LEARNING_RATE = 1e-4 # (0.0001)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1024
    NUM_EPOCHS = 5000
    if DEVICE == "cuda":
        NUM_WORKERS = 4
    else:
        NUM_WORKERS = 0
    PIN_MEMORY = True
    FILE_NAME = "data_charger_energy_EVs_cleaned.csv"
    TARGET_COL = 'total_capacity'
    MASTER_KEY = "PC6"
    EXCLUDE_COLS = ["number_of_charger"]

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
    df = pd.read_csv(FILE_NAME)

    # target
    y = df[TARGET_COL]

    # feature -> all columns except the target and master key
    # concatenate the list of columns to exclude with the master key
    if EXCLUDE_COLS[0] is not None:
        EXCLUDE_COLS = EXCLUDE_COLS + [MASTER_KEY, TARGET_COL]
    else:
        EXCLUDE_COLS = [MASTER_KEY, TARGET_COL]
    X = df.drop(EXCLUDE_COLS, axis=1)

    # split into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=RANDOM_SEED)

    # Normalize the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # impute missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # specify input and output sizes
    INPUT_SIZE = X_train.shape[1]
    OUTPUT_SIZE = 1

    ############################
    # create data loaders
    ############################

    # import data loader
    from torch.utils.data import DataLoader
    from dataset import CustomDataset

    # create training data loader
    train_dataset = CustomDataset(X_train, y_train.to_numpy())
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )

    # create testing data loader
    test_dataset = CustomDataset(X_test, y_test.to_numpy())
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # Unit test the data class and data loader
    from utils import test_tensor_shapes
    test_tensor_shapes(train_data_loader, test_data_loader, INPUT_SIZE)

    ############################
    # create model
    ############################
    # instantiate model
    from model import FullyConnectedModel
    NUM_HIDDEN_LAYERS = 15
    NODES_PER_LAYER = 40

    model = FullyConnectedModel(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        nodes_per_layer=NODES_PER_LAYER,
        # ToDo: RMSE seems to have an error when dropout is used
        dropout_rate=0.2
    )
    model.to(DEVICE)

    # loss function -> RMSE
    from torch import nn
    loss_fn = nn.MSELoss(reduction='mean')

    # optimizer -> Adam
    from torch import optim
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # scheduler -> cosine annealing with warm restarts
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(NUM_EPOCHS*len(train_data_loader)*0.05),
        T_mult=2,
        eta_min=LEARNING_RATE * 1e-4,
    )

    # print model summary using torchsummary
    from torchinfo import summary
    summary(model, input_size=(INPUT_SIZE,), device=DEVICE)

    # print
    print(f"Device: {DEVICE}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Feature names: {X.columns.to_list()}")
    print(f"Number of training samples: {X_train.shape[0]}")
    print(f"Number of testing samples: {X_test.shape[0]}")

    ############################
    # train model
    ############################
    # import train function
    from train import train_fn, eval_fn
    from utils import save_checkpoint
    from tqdm import trange

    best_test_loss = float('inf')  # initialize with a high value

    # Train and evaluate the model
    progress_bar = trange(NUM_EPOCHS, desc="Training")
    for epoch in progress_bar:
        # training
        train_loss = train_fn(train_data_loader, model, loss_fn, DEVICE, optimizer, scheduler)

        # testing
        test_loss, mse, rmse, mae = eval_fn(test_data_loader, model, loss_fn, DEVICE)

        # Update the progress bar with the current epoch loss
        progress_bar.set_postfix({"train_loss": f"{train_loss:.4f}"})

        if epoch % 10 == 0:
            # print training loss and test metrics:
            print(
                f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Test Loss:{test_loss:.4f}, Test RMSE: {rmse:.2f}, Test MAE: {mae:.2f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save the best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            model_path = save_checkpoint(checkpoint, model_name="best_model")

if __name__ == '__main__':
    main()