import os
import torch
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

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

# Warmup Scheduler
# code taken from: https://www.kaggle.com/datasets/aryankhatana/pytorch-warmup-scheduler
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None, is_batch=False):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.is_batch = is_batch
        super(GradualWarmupScheduler, self).__init__(optimizer)

        # Add the _last_lr attribute to store the last learning rate values
        self._last_lr = self.get_lr()

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

    def get_last_lr(self):
        self._last_lr = self.get_lr()
        return self._last_lr