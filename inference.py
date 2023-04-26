import torch
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import FullyConnectedModel
from train import eval_fn
from utils import load_checkpoint

def main():
    # Set the model parameters
    RANDOM_SEED = 42
    NUM_HIDDEN_LAYERS = 4
    NODES_PER_LAYER = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TARGET_COL = 'total_capacity'
    MASTER_KEY = 'zipcode'

    ############################
    # set seeds
    ############################
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = False

    ############################
    # Load data
    ############################
    # Load data as in the main script
    df = pd.read_csv('charger_df_cleaned.csv')
    y = df[TARGET_COL]
    X = df.drop([TARGET_COL], axis=1)
    X_test, y_test = X, y

    # specify input and output sizes
    INPUT_SIZE = X_test.shape[1] - 1
    OUTPUT_SIZE = 1

    ############################
    # Create data loaders
    ############################
    # Create the test_data_loader without dropping the MASTER_KEY column
    test_dataset = CustomDataset(X_test.drop([MASTER_KEY], axis=1), y_test)
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
    )

    ############################
    # Load model
    ############################
    MODEL_NAME = "best_model"
    NUM_HIDDEN_LAYERS = 4
    NODES_PER_LAYER = 10

    model = FullyConnectedModel(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        nodes_per_layer=NODES_PER_LAYER
    )
    model.to(DEVICE)
    model, _ = load_checkpoint(MODEL_NAME, model)

    # Evaluate the model
    loss_fn = torch.nn.MSELoss(reduction='mean')
    test_loss, mse, rmse, mae = eval_fn(test_data_loader, model, loss_fn, DEVICE)

    print(f"Test Loss: {test_loss:.4f}, Test MSE: {mse:.2f}, Test RMSE: {rmse:.2f}, Test MAE: {mae:.2f}")

    # Generate predictions
    model.eval()
    y_pred = []
    with torch.no_grad():
        for X_batch, _ in test_data_loader:
            X_batch = X_batch.to(DEVICE)
            output = model(X_batch)
            y_pred.extend(output.cpu().numpy().flatten())

    # Add MASTER_KEY column to results DataFrame
    # Concatenate X_test, y_test, and y_pred in the same DataFrame
    X_test_y_test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    results = pd.concat([X_test_y_test, pd.Series(y_pred, name="y_pred")], axis=1)

    # Save results to CSV
    results.to_csv(os.path.join("models", f"{MODEL_NAME}_predictions.csv"), index=False)

    # Plot predictions
    plt.plot(X_test[MASTER_KEY], y_test, label="True Values", linestyle="-")
    plt.plot(X_test[MASTER_KEY], y_pred, label="Predicted Values", linestyle="--")
    plt.xlabel("X Values (Master Key)")
    plt.ylabel("Y Values (Total Capacity)")
    plt.title("True vs. Predicted Values")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()