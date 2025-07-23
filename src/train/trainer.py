import os
import sys
# Ensure the models directory is in the Python path
models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
print(f"Adding models path to sys.path: {models_path}")
if models_path not in sys.path:
    sys.path.append(models_path)

# import unet
import unet_model as unet
from tqdm import tqdm
import torch
import torch.nn as nn
import json
from datetime import datetime

# Define the directory to save the model
SAVE_DIR = 'models/'

# function train the model
def train_model(
    model,
    train_loader,
    epochs,
    optimizer_fn,
    optimizer_kwargs,
    loss_fn,
    val_loader=None,
    scheduler_fn=None,
    scheduler_kwargs=None,
    device=None,
    save_dir="checkpoints"
):
    """
    Train the model with the given parameters.
    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        epochs (int): Number of epochs to train.
        optimizer_fn (callable): Optimizer function (e.g., torch.optim.Adam).
        optimizer_kwargs (dict): Keyword arguments for the optimizer.
        loss_fn (callable): Loss function (e.g., nn.MSELoss).
        scheduler_fn (callable, optional): Learning rate scheduler function.
        scheduler_kwargs (dict, optional): Keyword arguments for the scheduler.
        device (torch.device, optional): Device to run the training on.
        save_dir (str, optional): Directory to save checkpoints and metadata.
    Returns:
        model (nn.Module): The trained model.
        losses (list): List of losses for each epoch.
    """

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")


    model.to(device)
    model.train()
    
    # initialise optimiser, loss, scheduler
    optimizer = optimizer_fn(model.parameters(), **optimizer_kwargs)
    criterion = loss_fn
    scheduler = None
    if scheduler_fn is not None:
        scheduler = scheduler_fn(optimizer, **scheduler_kwargs)

    losses = []

    # create checkpoint directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    for epoch in range(epochs):
        print(f"📦 Running epoch {epoch+1}")
        running_loss = 0.0

        # wrap the train_loader in tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", dynamic_ncols=True)
        # iterate over the train_loader

        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update the progress bar description with the current loss
            pbar.set_postfix(loss=loss.item())

        loss = running_loss / len(train_loader)

        losses.append(loss)

        # Step the scheduler if it exists
        if scheduler is not None:
            scheduler.step()

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            print(f'Validation Loss: {val_loss:.4f}')
            model.train()
        else:
            val_loss = None
        # Print validation loss if available
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}' if val_loss is not None else f'Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}')

    # save checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(SAVE_DIR, f'model_checkpoint_{timestamp}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs,
        'loss_history': losses
    }, checkpoint_path)
    
    # save the meta data
    metadata = {
        'model_name': model.__class__.__name__,
        'epochs': epochs,
        'optimizer': optimizer_fn.__name__,
        'optimizer_kwargs': optimizer_kwargs,
        'loss_fn': loss_fn.__name__,
        'scheduler': scheduler_fn.__name__ if scheduler_fn else None,
        'scheduler_kwargs': scheduler_kwargs if scheduler_kwargs else None,
        'device': str(device),
        'timestamp': timestamp,
        'loss_history': losses,
        'checkpoint_path': checkpoint_path
    }

    metadata_path = os.path.join(SAVE_DIR, f"run_metadata_{timestamp}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    

    print(f"Model and metadata saved to {SAVE_DIR}")
    print("Training complete.")
    # return the trained model and losses
    return model, losses

