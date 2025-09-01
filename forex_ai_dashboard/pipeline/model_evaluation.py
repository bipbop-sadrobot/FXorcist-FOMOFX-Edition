import torch
import torch.nn as nn
from forex_ai_dashboard.pipeline.evaluation_metrics import Accuracy, Precision, Recall, F1Score
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import argparse  # For configurable runs
import os
import zipfile

class LSTMModel(nn.Module):
    """LSTM model for binary direction prediction (up/down)."""
    def __init__(self, input_size=4, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Last timestep
        return self.sigmoid(out).squeeze(-1)  # Probabilities for binary classification

class ForexDataset(Dataset):
    """Custom Dataset for Forex time series with sequencing and direction labels."""
    def __init__(self, csv_file, seq_len=60):
        if csv_file.endswith('.zip'):
            try:
                with zipfile.ZipFile(csv_file, 'r') as zip_ref:
                    # Assuming there's only one CSV file in the zip
                    csv_name = zip_ref.namelist()[0]
                    with zip_ref.open(csv_name) as f:
                        self.data = pd.read_csv(f)
            except Exception as e:
                print(f"Error reading zip file: {e}")
                self.data = pd.DataFrame()  # Create an empty DataFrame
        else:
            self.data = pd.read_csv(csv_file)

        if self.data.empty:
            print("No data loaded. Check the CSV file or zip archive.")
            self.X, self.y = np.array([]), np.array([])
            return

        # Preprocessing: Select OHLC features
        self.features = self.data[['Open', 'High', 'Low', 'Close']]
        
        # Create binary labels if not present: 1 if next Close > current Close, else 0
        if 'Label' not in self.data.columns:
            self.data['Label'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(float)
            self.data = self.data.dropna()  # Drop last row without label
        
        # Scale features
        self.scaler = MinMaxScaler()
        scaled_features = self.scaler.fit_transform(self.features)
        
        # Create sequences
        self.X, self.y = [], []
        for i in range(len(scaled_features) - seq_len):
            self.X.append(scaled_features[i:i + seq_len])
            self.y.append(self.data['Label'].values[i + seq_len - 1])  # Label after sequence
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float), torch.tensor(self.y[idx], dtype=torch.float)

def evaluate_model(model, dataloader, device='cpu'):
    """
    Evaluates the model on the dataloader using classification metrics.

    Args:
        model: The trained LSTM model.
        dataloader: DataLoader for evaluation.
        device: 'cuda' if GPU available, else 'cpu'.

    Returns:
        Dictionary of metrics.
    """
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    accuracy = Accuracy()
    precision = Precision()
    recall = Recall()
    f1score = F1Score()
    with torch.no_grad():
        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            preds = model(features)
            preds = torch.round(preds)  # Binary predictions
            accuracy.update(preds, targets)
            precision.update(preds, targets)
            recall.update(preds, targets)
            f1score.update(preds, targets)
    return {
        "accuracy": accuracy.compute(),
        "precision": precision.compute(),
        "recall": recall.compute(),
        "f1score": f1score.compute(),
    }

def load_data(csv_file, batch_size=32, seq_len=60):
    """
    Loads and prepares DataLoader.

    Args:
        csv_file: Path to CSV.
        batch_size: Batch size.
        seq_len: Sequence length for LSTM.

    Returns:
        DataLoader.
    """
    dataset = ForexDataset(csv_file, seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)  # No shuffle for time series eval

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate LSTM on Forex data.')
    parser.add_argument('--csv_file', type=str, default='data/data/raw/ejtrader_eurusd_m1.csv', help='Path to CSV.')
    parser.add_argument('--model_path', type=str, default='lstm_model.pth', help='Path to trained model weights.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--seq_len', type=int, default=60, help='Sequence length.')
    args = parser.parse_args()

    # Load data
    dataloader = load_data(args.csv_file, args.batch_size, args.seq_len)

    # Load model
    model = LSTMModel(input_size=4, hidden_size=50, num_layers=2)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))  # Load trained weights
        print("Loaded model weights from", args.model_path)
    else:
        print("Model weights file not found. Using randomly initialized model.")

    # Evaluate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics = evaluate_model(model, dataloader, device)

    # Print metrics
    print(metrics)
