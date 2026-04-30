import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
 
 
def detection_metrics(
    y_true:     np.ndarray,
    y_pred_prob: np.ndarray,
    threshold:  float = 0.5,
) -> dict:
    y_pred = (y_pred_prob >= threshold).astype(int)
 
    cm  = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
 
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        auc = float("nan")
 
    return {
        "accuracy":         acc,
        "precision":        pre,
        "recall":           rec,
        "f1":               f1,
        "auc_roc":          auc,
        "confusion_matrix": cm,
    }
 
 
def print_detection_metrics(metrics: dict) -> None:
    print("\n── Detection Metrics ──────────────────────")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
    print("────────────────────────────────────────────\n")
 
  
def magnitude_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    residuals = y_pred - y_true
    mae  = np.abs(residuals).mean()
    rmse = np.sqrt(np.mean(residuals ** 2))
 
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2     = 1.0 - ss_res / (ss_tot + 1e-12)
 
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}
 
 
def print_magnitude_metrics(metrics: dict) -> None:
    print("\n── Magnitude Metrics ──────────────────────")
    print(f"  MAE  : {metrics['mae']:.4f}")
    print(f"  RMSE : {metrics['rmse']:.4f}")
    print(f"  R²   : {metrics['r2']:.4f}")
    print("────────────────────────────────────────────\n")
 
 
_R_EARTH_KM = 6371.0   
 
def haversine_distance(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: np.ndarray, lon2: np.ndarray,
) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a    = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2.0 * _R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
 
 
def location_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    lat_mae   = float(np.abs(y_pred[:, 0] - y_true[:, 0]).mean())
    lon_mae   = float(np.abs(y_pred[:, 1] - y_true[:, 1]).mean())
    depth_mae = float(np.abs(y_pred[:, 2] - y_true[:, 2]).mean())
 
    dist_km   = haversine_distance(
        y_true[:, 0], y_true[:, 1],
        y_pred[:, 0], y_pred[:, 1],
    )
    mean_dist_km   = float(dist_km.mean())
    median_dist_km = float(np.median(dist_km))
 
    return {
        "lat_mae":        lat_mae,
        "lon_mae":        lon_mae,
        "depth_mae_km":   depth_mae,
        "mean_dist_km":   mean_dist_km,
        "median_dist_km": median_dist_km,
    }
 
 
def print_location_metrics(metrics: dict) -> None:
    print("\n── Location Metrics ───────────────────────")
    print(f"  Lat  MAE   : {metrics['lat_mae']:.4f}°")
    print(f"  Lon  MAE   : {metrics['lon_mae']:.4f}°")
    print(f"  Depth MAE  : {metrics['depth_mae_km']:.2f} km")
    print(f"  Mean  dist : {metrics['mean_dist_km']:.2f} km")
    print(f"  Median dist: {metrics['median_dist_km']:.2f} km")
    print("────────────────────────────────────────────\n")
  
def full_metrics_report(
    det_true, det_prob,
    mag_true, mag_pred,
    loc_true, loc_pred,
) -> dict:
    det = detection_metrics(det_true, det_prob)
    mag = magnitude_metrics(mag_true, mag_pred)
    loc = location_metrics(loc_true, loc_pred)
 
    print_detection_metrics(det)
    print_magnitude_metrics(mag)
    print_location_metrics(loc)
 
    return {"detection": det, "magnitude": mag, "location": loc}
 
 
if __name__ == "__main__":
    N = 200
    det_true  = np.random.randint(0, 2, N)
    det_prob  = np.clip(det_true + np.random.randn(N) * 0.3, 0, 1)
    mag_true  = np.random.uniform(2, 7, N)
    mag_pred  = mag_true + np.random.randn(N) * 0.5
    loc_true  = np.column_stack([
        np.random.uniform(-90, 90, N),
        np.random.uniform(-180, 180, N),
        np.random.uniform(0, 100, N),
    ])
    loc_pred  = loc_true + np.random.randn(N, 3) * [1, 1, 5]
 
    full_metrics_report(det_true, det_prob, mag_true, mag_pred, loc_true, loc_pred)
    print("Metrics self-test passed ")