import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from config import MODEL_CONFIG, TRAIN_CONFIG, PATHS
from data.collector import generate_synthetic_data
from data.preprocessor import process_snapshots
from data.labeler import compute_labels, align_data_and_labels
from model.deeplob import DeepLOB
from model.export import export_model

def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Calculate inverse frequency weights to handle class imbalance."""
    counts = np.bincount(labels, minlength=MODEL_CONFIG.num_classes)
    # Avoid division by zero
    counts[counts == 0] = 1 
    
    weights = 1.0 / counts
    # Normalize to sum to num_classes
    weights = weights / weights.sum() * MODEL_CONFIG.num_classes
    return torch.tensor(weights, dtype=torch.float32)

def prepare_data():
    print("Preparing data...")
    # Generate synthetic data since we don't have historical data yet
    raw_snapshots = generate_synthetic_data(num_samples=10000, num_levels=MODEL_CONFIG.num_levels)
    
    # Preprocess (Normalization)
    # shape: (N - lookback + 1, 1, lookback, 40)
    tensors = process_snapshots(raw_snapshots, num_levels=MODEL_CONFIG.num_levels, lookback=MODEL_CONFIG.lookback)
    
    # Label
    # shape: (N,)
    raw_labels = compute_labels(raw_snapshots, horizon=TRAIN_CONFIG.horizon, alpha=TRAIN_CONFIG.alpha)
    
    # Align tensors and labels
    tensors, labels = align_data_and_labels(tensors.numpy(), raw_labels, lookback=MODEL_CONFIG.lookback)
    
    # We must discard the last `horizon` elements because we can't label the end of the sequence
    valid_len = len(labels) - TRAIN_CONFIG.horizon
    tensors = torch.from_numpy(tensors[:valid_len])
    labels = torch.from_numpy(labels[:valid_len])
    
    print(f"Dataset shape: {tensors.shape}, Labels shape: {labels.shape}")
    
    # Temporal Split: 70% Train, 15% Val, 15% Test
    n = len(labels)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    X_train, y_train = tensors[:train_end], labels[:train_end]
    X_val, y_val = tensors[train_end:val_end], labels[train_end:val_end]
    X_test, y_test = tensors[val_end:], labels[val_end:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def train():
    device = torch.device(TRAIN_CONFIG.device if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data()
    
    # Datasets & Loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG.batch_size, shuffle=False) # Important: Don't shuffle time series easily, though here it's independent windows. Actually, shuffling train is OK to break correlation in batches, but let's keep false or careful true. Let's shuffle train.
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG.batch_size, shuffle=False)
    
    # Model
    model = DeepLOB(
        num_classes=MODEL_CONFIG.num_classes,
        conv_channels=MODEL_CONFIG.conv_channels,
        inception_channels=MODEL_CONFIG.inception_channels,
        lstm_hidden=MODEL_CONFIG.lstm_hidden
    ).to(device)
    
    # Loss & Optimizer
    # Handle Class Imbalance
    class_weights = get_class_weights(y_train.numpy()).to(device)
    print(f"Computed class weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG.learning_rate, weight_decay=TRAIN_CONFIG.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(TRAIN_CONFIG.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
        scheduler.step()
        
        train_loss /= len(train_loader.dataset)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_acc = 100. * correct / total
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Early stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(PATHS.models_dir, "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CONFIG.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    # Load best model for export
    print("Loading best model for export...")
    model.load_state_dict(torch.load(os.path.join(PATHS.models_dir, "best_model.pt")))
    
    safetensors_path = os.path.join(PATHS.models_dir, "deeplob.safetensors")
    config_path = os.path.join(PATHS.models_dir, "model_config.json")
    export_model(model, safetensors_path, config_path)
    print(f"Pipeline complete. Safetensors ready at: {safetensors_path}")

if __name__ == "__main__":
    train()
