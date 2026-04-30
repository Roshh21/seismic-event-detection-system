import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
 
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
 
class ResConvBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = 5, pool: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel,
                               padding=kernel // 2, bias=False)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel,
                               padding=kernel // 2, bias=False)
        self.bn2   = nn.BatchNorm1d(channels)
        self.pool  = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
 
    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.gelu(out + residual)
        return self.pool(out)
 
 
class ConvStem(nn.Module): 
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_ch, 32,      kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.MaxPool1d(2),                                      # /2
 
            nn.Conv1d(32, 64,         kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.MaxPool1d(2),                                      # /4
 
            nn.Conv1d(64, 128,        kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(2),                                      # /8
 
            nn.Conv1d(128, out_ch,    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.MaxPool1d(2),                                      # /16
        )
 
    def forward(self, x):
        return self.layers(x)
 
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
 
    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
  
class LocationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int   = 3,
        d_model:     int   = 192,
        nhead:       int   = 6,
        num_layers:  int   = 4,
        dropout:     float = 0.25,
    ):
        super().__init__()
 
        self.stem = ConvStem(in_channels, d_model)
 
        self.res_blocks = nn.Sequential(
            ResConvBlock(d_model, kernel=5, pool=1),
            ResConvBlock(d_model, kernel=5, pool=1),
        )
 
        self.pos_enc = PositionalEncoding(d_model, max_len=2000, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
 
        self.head = nn.Sequential(
            nn.Linear(d_model * 2 + 6, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),   # [lat, lon, depth]
        )
 
    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)           # (B, C, T)
        x = self.stem(x)                  # (B, d_model, seq)
        x = self.res_blocks(x)
        x = x.permute(0, 2, 1)           # (B, seq, d_model)
 
        x = self.pos_enc(x)
        x = self.transformer(x)           # (B, seq, d_model)
 
        mean_pool = x.mean(dim=1)         # (B, d_model)
        max_pool  = x.max(dim=1).values   # (B, d_model)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        pooled = torch.cat([pooled, meta], dim=-1)
 
        return self.head(pooled)           # (B, 3)
 
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
 
 
def train_location_model(
    X_train: np.ndarray,
    meta_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    meta_val: np.ndarray,
    y_val: np.ndarray,
    epochs:       int   = 40,
    batch_size:   int   = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience:     int   = 8,
) -> tuple:
    device = get_device()
    print(f"[Location] Training on {device}  |  "
          f"Train={len(X_train)}, Val={len(X_val)}, Epochs={epochs}")
 
    def make_loader(X, meta, y, shuffle):
        ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(meta).float(),
        torch.from_numpy(y).float(),
    )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
 
    train_loader = make_loader(X_train, meta_train, y_train, True)
    val_loader   = make_loader(X_val, meta_val, y_val, False)
 
    model     = LocationPredictor().to(device)
    def weighted_location_loss(pred, target):
        lat_loss   = ((pred[:, 0] - target[:, 0]) ** 2) * 2.0
        lon_loss   = ((pred[:, 1] - target[:, 1]) ** 2) * 2.0
        depth_loss = ((pred[:, 2] - target[:, 2]) ** 2) * 0.75
        return (lat_loss + lon_loss + depth_loss).mean()

    criterion = weighted_location_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
 
    history   = {"loss": [], "val_loss": []}
    best_val  = float("inf")
    wait      = 0
    best_state = None
 
    for epoch in range(1, epochs + 1):
        model.train()
        sum_loss, total = 0.0, 0

        for xb, mb, yb in train_loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(xb, mb)
            loss = criterion(pred, yb)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            sum_loss += loss.item() * len(xb)
            total += len(xb)

        train_loss = sum_loss / total

        model.eval()
        v_loss, v_total = 0.0, 0

        with torch.no_grad():
            for xb, mb, yb in val_loader:
                xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)

                pred = model(xb, mb)

                v_loss += criterion(pred, yb).item() * len(xb)
                v_total += len(xb)

        val_loss = v_loss / v_total
        scheduler.step(val_loss)

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"  Epoch {epoch:03d}/{epochs} | "
            f"train_MSE={train_loss:.4f} | "
            f"val_MSE={val_loss:.4f}"
        )

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            wait = 0
            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
            torch.save(
                best_state,
                str(CHECKPOINT_DIR / "location_best.pt")
            )
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break
 
    if best_state:
        model.load_state_dict(best_state)
    model.to("cpu")
    return model, history
  
def predict_location(
    model: LocationPredictor,
    X:     np.ndarray,
    meta,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size])
            mb = torch.from_numpy(meta[i:i+batch_size]).float()
            preds.append(model(xb, mb).numpy())
    return np.concatenate(preds, axis=0)
 
 
def load_location_model(path: str = None) -> LocationPredictor:
    path  = path or str(CHECKPOINT_DIR / "location_best.pt")
    model = LocationPredictor()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"[Location] Loaded model from {path}")
    return model
 
 
if __name__ == "__main__":
    model        = LocationPredictor()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LocationPredictor  |  Parameters: {total_params:,}")
 
    dummy = torch.randn(4, 6000, 3)
    dummy_meta = torch.randn(4, 6)

    out = model(dummy, dummy_meta)
    print(f"Output shape: {out.shape}  (expected: [4, 3])")
    print("Location model smoke-test passed ✓")