import torch
import numpy as np
import matplotlib.pyplot as plt
import time as T
from torch.cuda.amp import GradScaler, autocast
from utils.evaluation import validate_model
import os
import math

def train_model(
    model, 
    num_epochs, 
    train_loader, 
    validate_loader, 
    criterion, 
    optimizer, 
    scheduler, 
    models_dir,
    images_dir,
    model_name,
    config_name,
    scaler, 
    early_stopping=None,
    device='cuda',
    week='5week',
    is_save=False
):
    train_losses = []
    validate_losses = []
    best_val_loss = np.inf

    # Create a directory for the current model if it doesn't exist
    model_save_dir = os.path.join(models_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    # Define paths for saving the best and last model
    best_model_path = os.path.join(model_save_dir, f'{model_name}_{config_name}_best.pth')
    last_model_path = os.path.join(model_save_dir, f'{model_name}_{config_name}_last.pth')

    # for epoch in range(num_epochs): 
    # 加上进度条
    from tqdm import tqdm
    for epoch in range(num_epochs):


        model.train()
        running_loss = 0.0
        start_time = T.time()
        
        for inputs, labels, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            # #检测inputs 和 labels是否为nan
            # if torch.isnan(inputs).any() or torch.isnan(labels).any():
            #     print('-------------inputs or labels is nan')
            #     break
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
            loss = criterion(outputs.float(), labels.float())  # 关键：loss 用 fp32
            # print(loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        

        # Validation step
        val_loss = validate_model(model, validate_loader, criterion,model_name,device,week)
        validate_losses.append(val_loss)
        epoch_duration = T.time() - start_time
        print(
            f'{model_name} Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, '
            f'Validate Loss: {val_loss:.4f}, Duration: {epoch_duration:.2f} sec'
        )
        if math.isnan(running_loss) or math.isnan(epoch_loss):
            print('-------------NaN detected')
            print(f'running_loss: {running_loss}')
            print(f'epoch_loss: {epoch_loss}')
            break
        # Save the last model every 5 epochs
        if (epoch + 1) % 5 == 0 and is_save:
            torch.save(model.state_dict(), last_model_path)

        # Save the best model
        if val_loss < best_val_loss:
            print(f'{model_name} Saving new best model at epoch {epoch+1}')
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        # Early stopping
        if early_stopping:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"{model_name} Early stopping")
                break
        if running_loss is None:
            print('-------------None')
            break

    # # Plot loss curves and save the figure
    # plt.figure(figsize=(10, 5))
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(validate_losses, label='Validate Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # loss_plot_path = os.path.join(images_dir, f'{model_name}_{config_name}_loss.png')
    # plt.savefig(loss_plot_path)

    return best_model_path  # Return the path of the best model
