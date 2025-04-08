import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from imblearn.over_sampling import SMOTE
import os
import datetime
import csv
import time

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_data():
    """
    Load and return a regression dataset.
    we'll use the California Housing dataset.
    """
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target
    return X, y


def preprocess_data(X, y):
    """
    Preprocess the data:
    1. Split into train/validation/test sets
    2. Scale features
    3. Apply oversampling to training data

    Returns preprocessed data splits.
    """
    # Split into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Further split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE for regression using target binning
    n_bins = 5
    y_binned = pd.qcut(y_train, n_bins, labels=False)
    smote = SMOTE(random_state=42)
    X_train_res, y_binned_res = smote.fit_resample(X_train_scaled, y_binned)

    # Calculate median y values for each bin
    bin_medians = pd.Series(y_train).groupby(y_binned).median().to_dict()
    y_train_res = np.array([bin_medians[b] for b in y_binned_res])

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_res, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_res, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor


class RegressionDataset(Dataset):
    """
    Dataset class for regression data.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create PyTorch DataLoaders for training and validation.
    """
    train_dataset = RegressionDataset(X_train, y_train)
    val_dataset = RegressionDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class RegressionModel(nn.Module):
    """
    Neural network model for regression.
    """

    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def add_l1_regularization(model, loss, lambda_l1=0.001):
    """
    Add L1 regularization to the loss function.
    """
    l1_reg = torch.tensor(0., requires_grad=True)
    for param in model.parameters():
        l1_reg = l1_reg + torch.sum(torch.abs(param))
    return loss + lambda_l1 * l1_reg


class CSVLogger:
    """
    Simple CSV logger for PyTorch training.
    """

    def __init__(self, filename):
        self.filename = filename
        self.rows = []

    def append(self, metrics):
        self.rows.append(metrics)

    def save(self):
        with open(self.filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)


def train_model(model, train_loader, val_loader, num_epochs=100):
    """
    Train the model and log the training details.
    Returns training history.
    """
    # Setup logging
    log_dir = os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    csv_logger = CSVLogger(os.path.join(log_dir, 'training.log'))

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    best_val_loss = float('inf')
    patience = 10
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_mae = 0.0

        # Training loop
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss = add_l1_regularization(model, loss)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_mae += mean_absolute_error(targets.cpu().detach().numpy(),
                                             outputs.cpu().detach().numpy()) * inputs.size(0)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                val_mae += mean_absolute_error(targets.cpu().numpy(),
                                               outputs.cpu().numpy()) * inputs.size(0)

        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_mae = train_mae / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_mae = val_mae / len(val_loader.dataset)

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)

        # Log metrics
        csv_logger.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'time': time.time() - start_time
        })

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model weights
    model.load_state_dict(torch.load(os.path.join(log_dir, 'best_model.pth')))
    csv_logger.save()

    return model, history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using multiple performance metrics.
    """
    model.eval()

    # Convert to tensors if needed
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=torch.float32)

    X_test, y_test = X_test.to(device), y_test.to(device)

    with torch.no_grad():
        y_pred = model(X_test)

    y_true = y_test.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

    print(f"Test Metrics:\n"
          f"MSE: {metrics['MSE']:.4f}\n"
          f"MAE: {metrics['MAE']:.4f}\n"
          f"R²: {metrics['R2']:.4f}")

    return metrics


def plot_training_history(history):
    """
    Plot the training and validation loss curves.
    """
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()

    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'], label='Train MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('MAE Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_plots.png')
    plt.show()


def main():
    """
    Main function to run the entire pipeline.
    """
    print("Loading data...")
    X, y = load_data()

    print("Preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(X, y)

    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val)

    print("Creating model...")
    input_size = X_train.shape[1]
    model = RegressionModel(input_size).to(device)
    print(model)

    print("Training model...")
    model, history = train_model(model, train_loader, val_loader)

    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    print("Plotting training history...")
    plot_training_history(history)

    print("Assignment completed!")


if __name__ == "__main__":
    main()
