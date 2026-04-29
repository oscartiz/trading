import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

from config import MODEL_CONFIG, TRAIN_CONFIG, PATHS
from data.collector import generate_synthetic_data
from data.preprocessor import ZScorePreprocessor
from data.labeler import compute_labels
from model.deeplob import DeepLOB
from model.export import export_model

def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Calculate inverse frequency weights to handle class imbalance dynamically from training set."""
    counts = np.bincount(labels, minlength=MODEL_CONFIG.num_classes)
    counts[counts == 0] = 1 # Avoid division by zero
    
    weights = 1.0 / counts
    weights = weights / weights.sum() * MODEL_CONFIG.num_classes
    return torch.tensor(weights, dtype=torch.float32)

def align_labels(raw_labels: np.ndarray, lookback: int) -> torch.Tensor:
    """Drops the first `lookback - 1` labels because the rolling window removes them."""
    return torch.from_numpy(raw_labels[lookback - 1:])

def prepare_data():
    print("Preparing data...")
    # Using 50k samples to simulate a realistic dataset for the pipeline test
    raw_snapshots = generate_synthetic_data(num_samples=50000, num_levels=MODEL_CONFIG.num_levels)
    
    # Label the entire dataset
    raw_labels = compute_labels(raw_snapshots, horizon=TRAIN_CONFIG.horizon, alpha=TRAIN_CONFIG.alpha)
    
    # Drop the end where we can't look forward
    valid_len = len(raw_labels) - TRAIN_CONFIG.horizon
    raw_snapshots = raw_snapshots[:valid_len]
    raw_labels = raw_labels[:valid_len]
    
    # Chronological Split (70/15/15)
    n = len(raw_labels)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    train_snaps, train_labels = raw_snapshots[:train_end], raw_labels[:train_end]
    val_snaps, val_labels = raw_snapshots[train_end:val_end], raw_labels[train_end:val_end]
    test_snaps, test_labels = raw_snapshots[val_end:], raw_labels[val_end:]
    
    # Z-Score Normalization (fit strictly on training set)
    print("Normalizing data (fitting on train split only to prevent leakage)...")
    preprocessor = ZScorePreprocessor(num_levels=MODEL_CONFIG.num_levels, lookback=MODEL_CONFIG.lookback)
    X_train = preprocessor.fit_transform(train_snaps)
    X_val = preprocessor.transform(val_snaps)
    X_test = preprocessor.transform(test_snaps)
    
    # Align labels with the windowed tensors
    y_train = align_labels(train_labels, MODEL_CONFIG.lookback)
    y_val = align_labels(val_labels, MODEL_CONFIG.lookback)
    y_test = align_labels(test_labels, MODEL_CONFIG.lookback)
    
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shapes: X={X_val.shape}, y={y_val.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def train():
    device = torch.device(TRAIN_CONFIG.device if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data()
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAIN_CONFIG.batch_size, shuffle=False)
    
    model = DeepLOB(
        num_classes=MODEL_CONFIG.num_classes,
        conv_channels=MODEL_CONFIG.conv_channels,
        inception_channels=MODEL_CONFIG.inception_channels,
        lstm_hidden=MODEL_CONFIG.lstm_hidden
    ).to(device)
    
    # Dynamic Class Weights
    class_weights = get_class_weights(y_train.numpy()).to(device)
    print(f"Computed Class Weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG.learning_rate, weight_decay=TRAIN_CONFIG.weight_decay)
    
    # PyTorch 2.x note: ReduceLROnPlateau verbose removed.
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Starting robust training loop...")
    for epoch in range(TRAIN_CONFIG.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation & Metrics Calculation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        
        # Scikit-Learn Metrics
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro F1: {macro_f1:.4f}")
        print("Confusion Matrix [Up(0), Down(1), Stationary(2)]:\n", cm)
        
        scheduler.step(val_loss)
        
        # Early stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(PATHS.models_dir, "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(" -> Saved new best model.")
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CONFIG.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    print("Loading best model for export...")
    model.load_state_dict(torch.load(os.path.join(PATHS.models_dir, "best_model.pt")))
    
    safetensors_path = os.path.join(PATHS.models_dir, "deeplob.safetensors")
    config_path = os.path.join(PATHS.models_dir, "model_config.json")
    export_model(model, safetensors_path, config_path)
    print(f"Pipeline complete. Safetensors ready at: {safetensors_path}")

if __name__ == "__main__":
    train()
