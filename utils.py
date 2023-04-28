import os
import torch

def test_tensor_shapes(train_data_loader, test_data_loader, input_size):
    for loader in [train_data_loader, test_data_loader]:
        for idx, (X, y) in enumerate(loader):
            assert X.shape[-1] == input_size, f"Expected input size: {input_size}, got: {X.shape[1]}"
            assert y.shape[1] == 1, f"Output tensor shape mismatch. Expected: 1, Found: {len(y.shape)}"
            if idx >= 2:
                break

def save_checkpoint(checkpoint, model_name):
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    torch.save(checkpoint, model_path)
    return model_path

def load_checkpoint(model_name, model, optimizer=None):
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, "models")
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer