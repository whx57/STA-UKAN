import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from utils.utils import EarlyStopping, set_seed, write_log, write_info, write_log_metrics
import lpips
import os
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

def validate_model(model, validate_loader, criterion, model_name, device, week):
    model.eval()
    val_running_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total_lpips = 0.0
    total_scc = 0.0  # Added for SCC
    total_r2 = 0.0   # Added for R²
    num_total_pixels = 0
    num_batches = 0

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    with torch.no_grad():
        for inputs, labels, masks in validate_loader:
            inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            # if masks.size(0) != outputs.size(0):
            #     # 情况1: masks是单张图像 (C,H,W)，需要添加batch维度
            #     if masks.dim() == 3:
            #         masks = masks.unsqueeze(0)  # 变为 (1,C,H,W)
                
            #     # 情况2: masks已有batch维度但为1，需要扩展
            #     if masks.size(0) == 1:
            #         batch_size = outputs.size(0)
            #         masks = masks.expand(batch_size, -1, -1, -1)  # 扩展batch维度
            if masks.dim() == 3:
                masks=masks.unsqueeze(1)  # 确保masks是四维的 (B,1,H,W)
            mask = ~masks.bool()
            outputs_flat = outputs[mask].cpu().numpy()  # Convert to numpy for SCC and R²
            labels_flat = labels[mask].cpu().numpy()
            mae = torch.abs(outputs[mask] - labels[mask]).mean().item()
            total_mae += mae * outputs_flat.size
            mse = torch.mean((outputs[mask] - labels[mask]) ** 2).item()
            total_mse += mse * outputs_flat.size
            num_valid_pixels = outputs_flat.size
            num_total_pixels += num_valid_pixels
            outputs_np = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            masks_np = masks.cpu().numpy().astype(bool)
            outputs_np_masked = outputs_np.copy()
            labels_np_masked = labels_np.copy()
            outputs_np_masked[masks_np] = 0
            labels_np_masked[masks_np] = 0
            batch_size = outputs_np.shape[0]
            #如果存在nan就跳过
            if np.isnan(outputs_np_masked).any() or np.isnan(labels_np_masked).any():
                print('-------------outputs or labels is nan')
                break
            for i in range(batch_size):
                ssim_val = ssim(outputs_np_masked[i, 0], labels_np_masked[i, 0], 
                               data_range=labels_np_masked[i, 0].max() - labels_np_masked[i, 0].min())
                total_ssim += ssim_val
                mse_value = np.mean((outputs_np_masked[i, 0] - labels_np_masked[i, 0]) ** 2)
                if mse_value == 0:
                    psnr_val = 100
                else:
                    psnr_val = 10 * np.log10((labels_np_masked[i, 0].max() ** 2) / mse_value)
                total_psnr += psnr_val
                outputs_masked = outputs[i:i+1] * (~masks[i:i+1])
                labels_masked = labels[i:i+1] * (~masks[i:i+1])
                lpips_val = loss_fn_alex(outputs_masked, labels_masked).item()
                total_lpips += lpips_val
                # SCC and R² for each sample

                scc_val, _ = spearmanr(outputs_np_masked[i, 0].flatten(), labels_np_masked[i, 0].flatten())
                # r2_val = r2_score(labels_np_masked[i, 0].flatten(), outputs_np_masked[i, 0].flatten())
                r2_val= 0
                total_scc += scc_val
                total_r2 += r2_val
            num_batches += batch_size
    if num_batches == 0:
        print("No valid batches found.")
        return 0
    avg_loss = val_running_loss / len(validate_loader.dataset)
    avg_mae = total_mae / num_total_pixels
    avg_mse = total_mse / num_total_pixels
    avg_ssim = total_ssim / num_batches
    avg_psnr = total_psnr / num_batches
    avg_lpips = total_lpips / num_batches
    avg_scc = total_scc / num_batches  # Average SCC
    avg_r2 = total_r2 / num_batches    # Average R²

    print(f'{week}-{model_name} Validation Metrics - Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, '
          f'MSE: {avg_mse:.4f}, SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}, LPIPS: {avg_lpips:.4f}, '
          f'SCC: {avg_scc:.4f}, R2: {avg_r2:.4f}')

    return avg_loss


def test_model(model, test_loader, criterion, model_name, config_name, logs_dir, device, week):
    model.eval()
    val_running_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total_lpips = 0.0
    total_scc = 0.0  # Added for SCC
    total_r2 = 0.0   # Added for R²
    num_total_pixels = 0
    num_batches = 0

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    with torch.no_grad():
        for inputs, labels, masks in test_loader:
            inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
            outputs = model(inputs)
            if masks.dim() == 3:
                masks=masks.unsqueeze(1)  # 确保masks是四维的 (B,1,H,W)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            mask = ~masks.bool()
            outputs_flat = outputs[mask].cpu().numpy()
            labels_flat = labels[mask].cpu().numpy()
            mae = torch.abs(outputs[mask] - labels[mask]).mean().item()
            total_mae += mae * outputs_flat.size
            mse = torch.mean((outputs[mask] - labels[mask]) ** 2).item()
            total_mse += mse * outputs_flat.size
            num_valid_pixels = outputs_flat.size
            num_total_pixels += num_valid_pixels
            outputs_np = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            masks_np = masks.cpu().numpy().astype(bool)
            outputs_np_masked = outputs_np.copy()
            labels_np_masked = labels_np.copy()
            outputs_np_masked[masks_np] = 0
            labels_np_masked[masks_np] = 0
            batch_size = outputs_np.shape[0]
            for i in range(batch_size):
                ssim_val = ssim(outputs_np_masked[i, 0], labels_np_masked[i, 0], 
                               data_range=labels_np_masked[i, 0].max() - labels_np_masked[i, 0].min())
                total_ssim += ssim_val
                mse_value = np.mean((outputs_np_masked[i, 0] - labels_np_masked[i, 0]) ** 2)
                if mse_value == 0:
                    psnr_val = 100
                else:
                    psnr_val = 10 * np.log10((labels_np_masked[i, 0].max() ** 2) / mse_value)
                total_psnr += psnr_val
                outputs_masked = outputs[i:i+1] * (~masks[i:i+1])
                labels_masked = labels[i:i+1] * (~masks[i:i+1])
                lpips_val = loss_fn_alex(outputs_masked, labels_masked).item()
                total_lpips += lpips_val
                # SCC and R² for each sample
                scc_val, _ = spearmanr(outputs_np_masked[i, 0].flatten(), labels_np_masked[i, 0].flatten())
                r2_val = r2_score(labels_np_masked[i, 0].flatten(), outputs_np_masked[i, 0].flatten())
                total_scc += scc_val
                total_r2 += r2_val
            num_batches += batch_size

    avg_loss = val_running_loss / len(test_loader.dataset)
    avg_mae = total_mae / num_total_pixels
    avg_mse = total_mse / num_total_pixels
    avg_ssim = total_ssim / num_batches
    avg_psnr = total_psnr / num_batches
    avg_lpips = total_lpips / num_batches
    avg_scc = total_scc / num_batches  # Average SCC
    avg_r2 = total_r2 / num_batches    # Average R²

    print(model_name)
    print(week)
    print(f'Test Metrics - Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, MSE: {avg_mse:.4f}, '
          f'SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}, LPIPS: {avg_lpips:.4f}, '
          f'SCC: {avg_scc:.4f}, R2: {avg_r2:.4f}')
    test_log_file = os.path.join(logs_dir, f'{week}_{model_name}_test_log.txt')
    # Update write_log_metrics to include SCC and R²
    write_log_metrics(f'Test {model_name}', avg_loss, avg_mae, avg_mse, avg_ssim, avg_psnr, avg_lpips, 
                      test_log_file, scc=avg_scc, r2=avg_r2)