import torch
import numpy as np
import random

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.val_loss_min = val_loss

def set_seed(seed_value=42,use_deterministic_algorithms=True):
    print(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(use_deterministic_algorithms)

    # """Set seed for reproducibility."""
    # random.seed(seed_value)       # Python random module
    # np.random.seed(seed_value)    # NumPy
    # torch.manual_seed(seed_value) # CPU
    # torch.cuda.manual_seed(seed_value)
    # torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def write_log(epoch, train_loss, val_loss, file_path):
    with open(file_path, 'a') as f:
        f.write(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validate Loss: {val_loss:.4f}\n')

def write_info(text, file_path):
    with open(file_path, 'a') as f:
        f.write(text)


def write_log_metrics(phase, loss, mae, mse, ssim_value, psnr_value, lpips_value, file_path,scc, r2):
    with open(file_path, 'a') as f:
        f.write(f'{phase} Metrics - Loss: {loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.4f}, LPIPS: {lpips_value:.4f},SCC{scc:.4f},R2{r2:.4f}\n')
