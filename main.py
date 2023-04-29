import torch
import pandas as pd
import os


def main():
    ############################
    # Hyperparameters
    ############################
    RANDOM_SEED = 42
    LEARNING_RATE = 5e-5 # (0.0001)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 4
    NUM_EPOCHS = 1000
    if DEVICE == "cuda":
        NUM_WORKERS = 2
    else:
        NUM_WORKERS = 0
    PIN_MEMORY = True
    LSTM = False

    # fetch data
    from data_preparation import prepare_data
    X_train, X_test, y_train, y_test, _ = prepare_data(pytorch=True)

    # convert y_train and y_test to numpy arrays
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # prepare data for LSTM
    from dataset import create_lookback_dataset
    if LSTM:
        LOOKBACK = 7
        X_train, y_train = create_lookback_dataset(X_train, y_train, LOOKBACK)
        X_test, y_test = create_lookback_dataset(X_test, y_test, LOOKBACK)
        print(f"Tensors reshaped for LSTM, new shape is: {X_train.shape, X_test.shape, y_train.shape, y_test.shape}")

    # specify input and output sizes
    INPUT_SIZE = X_train.shape[-1]
    OUTPUT_SIZE = 1

    ############################
    # create data loaders
    ############################
    # import data loader
    from torch.utils.data import DataLoader
    from dataset import CustomDataset

    # create training data loader
    train_dataset = CustomDataset(X_train, y_train)
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )

    # create testing data loader
    test_dataset = CustomDataset(X_test, y_test)
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
    # FC model
    from model import MLP
    NUM_HIDDEN_LAYERS = 8
    NODES_PER_LAYER = 300

    model = MLP(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        nodes_per_layer=NODES_PER_LAYER,
        dropout_rate=0.05
    )
    model.to(DEVICE)

    # MLP with BatchNorm
    from model import MLPWithBatchNorm
    NUM_HIDDEN_LAYERS = 8
    NODES_PER_LAYER = 300
    model = MLPWithBatchNorm(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        nodes_per_layer=NODES_PER_LAYER,
        dropout_rate=0.05
    )
    model.to(DEVICE)

    # Halfing model
    # from model import HalfingModel
    # model = HalfingModel(
    #     input_size=INPUT_SIZE,
    #     output_size=OUTPUT_SIZE,
    #     factor=20,
    #     num_blocks=3,
    #     dropout_rate=0
    # )
    # model.to(DEVICE)


    # LSTM
    if LSTM:
        from model import LSTM1
        model = LSTM1(input_size=INPUT_SIZE,
                      hidden_size=4,
                      num_stacked_layers=1,
                      device=DEVICE)
        model.to(DEVICE)

    ############################
    # Loss, optimizer, scheduler
    ############################

    # loss function -> RMSE
    from torch import nn
    loss_fn = nn.MSELoss(reduction='mean')

    # optimizer -> Adam
    from torch import optim
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    # scheduler -> cosine annealing with warm restarts
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(NUM_EPOCHS*len(train_data_loader)*0.03),
        T_mult=2,
        eta_min=LEARNING_RATE * 1e-4,
    )

    # GradualWarmupScheduler
    from utils import GradualWarmupScheduler
    WARMUP_EPOCHS = int(NUM_EPOCHS*len(train_data_loader)*0.03)
    scheduler = GradualWarmupScheduler(optimizer,
                                       multiplier=1,
                                       total_epoch=WARMUP_EPOCHS, # when to stop warmup
                                       after_scheduler=scheduler,
                                       is_batch = True)

    # print model summary using torchsummary
    from torchinfo import summary
    tensor_shape = (BATCH_SIZE, LOOKBACK, INPUT_SIZE) if LSTM else (BATCH_SIZE, INPUT_SIZE)
    summary(model, input_size=tensor_shape, device=DEVICE)  # Update the input_size to (BATCH_SIZE, INPUT_SIZE)

    # print
    print(f"Device: {DEVICE}")

    ############################
    # train model
    ############################
    # import train function
    from train import train_fn, eval_fn
    from utils import save_checkpoint
    from tqdm import trange

    best_test_loss = float('inf')  # initialize with a high value

    # Initialize an empty DataFrame for storing metrics
    metrics_df = pd.DataFrame(columns=["epoch", "train_loss", "test_loss", "test_rmse", "test_mae"])

    # Train and evaluate the model
    progress_bar = trange(NUM_EPOCHS)
    for epoch in progress_bar:
        # training
        train_loss = train_fn(train_data_loader, model, loss_fn, DEVICE, optimizer, scheduler)

        # testing
        test_loss, mse, rmse, mae = eval_fn(test_data_loader, model, loss_fn, DEVICE)

        # Update the progress bar with the current epoch loss
        progress_bar.set_postfix({"train_loss": f"{format(train_loss, ',.2f')}"})

        # log metrics
        data = pd.DataFrame(
            {"epoch": [epoch], "train_loss": [train_loss], "test_loss": [test_loss], "test_rmse": [rmse],
             "test_mae": [mae]})
        metrics_df = pd.concat([metrics_df, data], ignore_index=True)

        if epoch % 10 == 0:
            # print training loss and test metrics:
            print(
                f"Epoch: {epoch}, Train Loss: {format(train_loss, ',.2f')}, Test Loss:{format(test_loss, ',.2f')}, Test RMSE: {format(rmse, ',.2f')}, LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Save metrics DataFrame to a CSV file
            cwd = os.getcwd()
            metrics_df.to_csv(os.path.join(cwd, 'models', 'metrics.csv'), index=False)

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