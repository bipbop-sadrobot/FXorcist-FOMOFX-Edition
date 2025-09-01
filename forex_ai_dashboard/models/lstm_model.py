import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..utils.logger import logger


def _get_device(prefer_mps: bool = True) -> torch.device:
    # macOS MPS or CUDA if available; otherwise CPU
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LSTMModel(nn.Module):
    """
    A lightweight, production-friendly LSTM forecaster with:
      - Proper device handling (CPU/CUDA/MPS)
      - Optional dropout
      - Scikit-like API: fit / predict / save / load
      - Single-step regression head (last-timestep output)

    X shape: [batch, seq_len, input_dim]
    y shape: [batch] or [batch, 1]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        prefer_mps: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.device = _get_device(prefer_mps=prefer_mps)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        final_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(final_dim, output_dim)

        self.to(self.device)
        logger.info(
            f"Initialized LSTMModel(layers={num_layers}, hidden={hidden_dim}, "
            f"bidirectional={bidirectional}) on device={self.device}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # last timestep
        out = self.fc(out)           # [B, output_dim]
        return out

    # ---------- Scikit-like API ----------
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        shuffle: bool = False,
        val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        patience: int = 5,
    ):
        """
        Train with simple early stopping on validation loss (if provided).
        """
        self.train()
        X = X.to(self.device)
        y = y.to(self.device).view(-1, self.output_dim)

        ds = TensorDataset(X, y)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        patience_left = patience

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for xb, yb in dl:
                optimizer.zero_grad(set_to_none=True)
                preds = self.forward(xb)
                loss = criterion(preds, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(ds)

            msg = f"[LSTM] epoch {epoch}/{epochs} train_mse={epoch_loss:.6f}"
            # Validation
            if val_data is not None:
                self.eval()
                with torch.no_grad():
                    Xv, yv = val_data
                    Xv = Xv.to(self.device)
                    yv = yv.to(self.device).view(-1, self.output_dim)
                    pv = self.forward(Xv)
                    vloss = criterion(pv, yv).item()
                msg += f" val_mse={vloss:.6f}"
                self.train()

                if vloss < best_val - 1e-7:
                    best_val = vloss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                    patience_left = patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        logger.info(msg + " (early stop)")
                        break
            logger.info(msg)

        if best_state is not None:
            self.load_state_dict(best_state)
            logger.info(f"[LSTM] Restored best validation weights (val_mse={best_val:.6f})")

        return self

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.eval()
        X = X.to(self.device)
        out = self.forward(X)
        return out.squeeze(-1).detach().cpu()

    # ---------- Persistence ----------
    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "config": self._config_dict()}, path)
        logger.info(f"Saved LSTM model to {path}")

    @classmethod
    def load(cls, path: str | Path, prefer_mps: bool = True):
        checkpoint = torch.load(path, map_location=_get_device(prefer_mps=prefer_mps))
        config = checkpoint.get("config", {})
        model = cls(**config, prefer_mps=prefer_mps)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        logger.info(f"Loaded LSTM model from {path}")
        return model

    # ---------- Helpers ----------
    def _config_dict(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
        }
