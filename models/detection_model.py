import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
 
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True) 
class ConvBlock(nn.Module): 
    def __init__(
        self,
        in_ch:   int,
        out_ch:  int,
        kernel:  int = 7,
        stride:  int = 1,
        pool:    int = 2,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel,
                              stride=stride, padding=kernel // 2, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
 
    def forward(self, x):
        return self.pool(F.gelu(self.bn(self.conv(x))))
 
 
class PositionalEncoding(nn.Module): 
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
 
        pe = torch.zeros(max_len, d_model)
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
 
  
class EarthquakeDetector(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        d_model:     int = 128,
        nhead:       int = 4,
        num_layers:  int = 3,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(in_channels, 32,  kernel=7, pool=2),   
            ConvBlock(32,          64,  kernel=7, pool=2),   
            ConvBlock(64,          128, kernel=5, pool=2),   
            ConvBlock(128,         d_model, kernel=3, pool=2),  
        )
 
        self.pos_enc = PositionalEncoding(d_model, max_len=4000, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)           
        x = self.cnn(x)                   
        x = x.permute(0, 2, 1)           
 
        x = self.pos_enc(x)
        x = self.transformer(x)           
 
        x = x.mean(dim=1)                 
        return self.head(x)               
 
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
 
 
def train_detection_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    epochs:      int   = 30,
    batch_size:  int   = 32,
    lr:          float = 1e-3,
    weight_decay: float = 1e-4,
    patience:    int   = 7,
) -> tuple:
    device = get_device()
    print(f"[Detection] Training on {device}  |  "
          f"Train={len(X_train)}, Val={len(X_val)}, Epochs={epochs}")
    def make_loader(X, y, shuffle):
        ds = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y).unsqueeze(1),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=device.type == "cuda")
 
    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)
 
    model     = EarthquakeDetector().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
 
    history   = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    best_val  = float("inf")
    wait      = 0
    best_state = None
 
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
 
            running_loss += loss.item() * len(xb)
            preds  = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == yb).sum().item()
            total   += len(xb)
 
        train_loss = running_loss / total
        train_acc  = correct / total
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                v_loss  += criterion(logits, yb).item() * len(xb)
                preds   = (torch.sigmoid(logits) >= 0.5).float()
                v_correct += (preds == yb).sum().item()
                v_total   += len(xb)
 
        val_loss = v_loss / v_total
        val_acc  = v_correct / v_total
 
        scheduler.step()
 
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
 
        print(f"  Epoch {epoch:03d}/{epochs} | "
              f"loss={train_loss:.4f}  acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
 
        if val_loss < best_val - 1e-4:
            best_val   = val_loss
            wait       = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, str(CHECKPOINT_DIR / "detection_best.pt"))
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break
 
    if best_state:
        model.load_state_dict(best_state)
    model.to("cpu")
    return model, history
  
def predict_detection(
    model:   EarthquakeDetector,
    X:       np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb    = torch.from_numpy(X[i: i + batch_size])
            logits = model(xb)
            probs.append(torch.sigmoid(logits).squeeze(1).numpy())
    return np.concatenate(probs)
 
 
def load_detection_model(path: str = None) -> EarthquakeDetector:
    path   = path or str(CHECKPOINT_DIR / "detection_best.pt")
    model  = EarthquakeDetector()
    state  = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print(f"[Detection] Loaded model from {path}")
    return model
 
 
if __name__ == "__main__":
    model = EarthquakeDetector()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EarthquakeDetector  |  Parameters: {total_params:,}")
 
    dummy_input = torch.randn(4, 6000, 3)
    out         = model(dummy_input)
    print(f"Output shape: {out.shape}  (expected: [4, 1])")
    print("Detection model smoke-test passed ✓")