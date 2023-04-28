import torch

def main():
    ############################
    # Hyperparameters
    ############################
    RANDOM_SEED = 42
    LEARNING_RATE = 1e-4 # (0.0001)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    NUM_EPOCHS = 5000
    if DEVICE == "cuda":
        NUM_WORKERS = 4
    else:
        NUM_WORKERS = 0
    PIN_MEMORY = True
    LSTM = False

    # fetch data
    from data_preparation import prepare_data
    X_train, X_test, y_train, y_test, _ = prepare_data()

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
    # instantiate model
    from model import FullyConnectedModel
    NUM_HIDDEN_LAYERS = 12
    NODES_PER_LAYER = 120

    model = FullyConnectedModel(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        nodes_per_layer=NODES_PER_LAYER,
        dropout_rate=0.3
    )
    model.to(DEVICE)

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
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)

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

    ############################
    # train model
    ############################
    # import train function
    from train import train_fn, eval_fn
    from utils import save_checkpoint
    from tqdm import trange

    best_test_loss = float('inf')  # initialize with a high value

    # Train and evaluate the model
    progress_bar = trange(NUM_EPOCHS)
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