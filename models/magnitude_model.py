import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
 
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
 
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, pool=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel,
                              padding=kernel // 2, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
 
    def forward(self, x):
        return self.pool(F.gelu(self.bn(self.conv(x))))
 
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        
 
    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
 
class MagnitudePredictor(nn.Module):
    def __init__(
        self,
        in_channels: int   = 3,
        psd_dim:     int   = 18,
        d_model:     int   = 128,
        nhead:       int   = 4,
        num_layers:  int   = 3,
        dropout:     float = 0.2,
    ):
        super().__init__()
 
        self.cnn = nn.Sequential(
            ConvBlock(in_channels, 32,      kernel=7, pool=2),
            ConvBlock(32,          64,      kernel=7, pool=2),
            ConvBlock(64,          128,     kernel=5, pool=2),
            ConvBlock(128,         d_model, kernel=3, pool=2),
        )
 
        self.pos_enc = PositionalEncoding(d_model, max_len=4000, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
 
        self.head = nn.Sequential(
            nn.Linear(d_model + 64, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),   
        )
        
        self.psd_branch = nn.Sequential(
            nn.Linear(psd_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
        )
 
    def forward(self, x: torch.Tensor, psd) -> torch.Tensor:
        """x: (B, T, C)  →  output: (B, 1)"""
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)

        psd_feat = self.psd_branch(psd)

        combined = torch.cat([x, psd_feat], dim=1)

        return self.head(combined)        
  
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
    
 
def train_magnitude_model(
    X_train: np.ndarray,
    psd_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    psd_val: np.ndarray,
    y_val:   np.ndarray,
    epochs:       int   = 40,
    batch_size:   int   = 32,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    patience:     int   = 8,
) -> tuple:
    device = get_device()
    print(f"[Magnitude] Training on {device}  |  "
          f"Train={len(X_train)}, Val={len(X_val)}, Epochs={epochs}")
 
    def make_loader(X, psd, y, shuffle):
        ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(psd).float(),
        torch.from_numpy(y).unsqueeze(1).float(),
    )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
 
    train_loader = make_loader(X_train, psd_train, y_train, True)
    val_loader   = make_loader(X_val, psd_val, y_val, False)
 
    model = MagnitudePredictor(psd_dim=psd_train.shape[1]).to(device)
    def weighted_mse(pred, target):
        weights = 1.0 + ((target.abs() + 1.0) ** 2) / 2.0
        return ((pred - target) ** 2 * weights).mean()
    criterion = weighted_mse
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
 
    history   = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}
    best_val  = float("inf")
    wait      = 0
    best_state = None
 
    for epoch in range(1, epochs + 1):
        model.train()
        sum_loss, sum_mae, total = 0.0, 0.0, 0
        for xb, psdb, yb in train_loader:
            xb, psdb, yb = xb.to(device), psdb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, psdb)
            loss  = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
 
            sum_loss += loss.item() * len(xb)
            sum_mae  += torch.abs(pred - yb).sum().item()
            total    += len(xb)
 
        train_loss = sum_loss / total
        train_mae  = sum_mae  / total
 
        model.eval()
        v_loss, v_mae, v_total = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, psdb, yb in val_loader:
                xb, psdb, yb = xb.to(device), psdb.to(device), yb.to(device)
                pred = model(xb, psdb)
                v_loss += criterion(pred, yb).item() * len(xb)
                v_mae  += torch.abs(pred - yb).sum().item()
                v_total += len(xb)
 
        val_loss = v_loss / v_total
        val_mae  = v_mae  / v_total
 
        scheduler.step(val_loss)
 
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["mae"].append(train_mae)
        history["val_mae"].append(val_mae)
 
        print(f"  Epoch {epoch:03d}/{epochs} | "
              f"MSE={train_loss:.4f}  MAE={train_mae:.4f} | "
              f"val_MSE={val_loss:.4f}  val_MAE={val_mae:.4f}")
 
        if val_loss < best_val - 1e-4:
            best_val   = val_loss
            wait       = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, str(CHECKPOINT_DIR / "magnitude_best.pt"))
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break
 
    if best_state:
        model.load_state_dict(best_state)
    model.to("cpu")
    return model, history
 
 
def predict_magnitude(
    model: MagnitudePredictor,
    X: np.ndarray,
    psd: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).float()
            psdb = torch.from_numpy(psd[i:i+batch_size]).float()

            pred = model(xb, psdb)
            preds.append(pred.squeeze(1).numpy())

    return np.concatenate(preds)
 
 
def load_magnitude_model(path: str = None) -> MagnitudePredictor:
    path  = path or str(CHECKPOINT_DIR / "magnitude_best.pt")
    model = MagnitudePredictor()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"[Magnitude] Loaded model from {path}")
    return model
 
 
if __name__ == "__main__":
    model       = MagnitudePredictor()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MagnitudePredictor  |  Parameters: {total_params:,}")
 
    dummy = torch.randn(4, 6000, 3)
    dummy_psd = torch.randn(4, 18)
    out = model(dummy, dummy_psd)
    print(f"Output shape: {out.shape}  (expected: [4, 1])")
    print("Magnitude model smoke-test passed ✓")
    
    
    