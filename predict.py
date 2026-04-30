import argparse
import json
import sys
import os
import numpy as np
import pandas as pd
 
sys.path.insert(0, os.path.dirname(__file__))
 
from models.detection_model  import load_detection_model,  predict_detection
from models.magnitude_model  import load_magnitude_model,  predict_magnitude
from models.location_model   import load_location_model,   predict_location
from alerts.alert_engine     import run_alert_pipeline
from utils.visualization     import plot_waveform, create_folium_alert_map
from data.data_loader import extract_psd_features
 
def run_inference(
    waveform:   np.ndarray,   # (1, 6000, 3)
    trace_name: str = "unknown",
    metadata=None,
    det_path:   str = None,
    mag_path:   str = None,
    loc_path:   str = None,
    save_dir:   str = "outputs",
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
 
    det_model = load_detection_model(det_path)
    mag_model = load_magnitude_model(mag_path)
    loc_model = load_location_model(loc_path)
 
    det_prob  = float(predict_detection(det_model, waveform)[0])
    is_eq     = det_prob >= 0.5
    print(f"\n[Predict] Detection  : {'EARTHQUAKE' if is_eq else 'NOISE'} "
          f"(P={det_prob:.4f})")
 
    psd_features = extract_psd_features(waveform[0])   # shape (18,)
    psd_features = psd_features[np.newaxis, ...]       # shape (1, 18)
    
    psd_mean = np.load("checkpoints/psd_mean.npy")
    psd_std  = np.load("checkpoints/psd_std.npy")

    psd_features = (psd_features - psd_mean) / psd_std

    mag_pred = float(predict_magnitude(
        mag_model,
        waveform,
        psd_features
    )[0])  
    
    print(f"[DEBUG] Raw normalized prediction: {mag_pred}")

    mag_mean = np.load("checkpoints/magnitude_mean.npy")
    mag_std  = np.load("checkpoints/magnitude_std.npy")

    print(f"[DEBUG] magnitude_mean: {mag_mean}")
    print(f"[DEBUG] magnitude_std : {mag_std}")

    magnitude = mag_pred * mag_std + mag_mean

    print(f"[DEBUG] Final denormalized magnitude: {magnitude}")
    
    magnitude = mag_pred * mag_std + mag_mean
    
    
    print(f"[Predict] Magnitude  : {magnitude:.3f}")
 
    if metadata is not None:
        receiver_raw = np.array([
        [
            metadata["receiver_latitude"],
            metadata["receiver_longitude"],
            metadata["receiver_elevation_m"],
            metadata["p_arrival_sample"],
            metadata["s_arrival_sample"],
            metadata["s_arrival_sample"] - metadata["p_arrival_sample"],
        ]
        ], dtype=np.float32)

        receiver_mean = np.load("checkpoints/receiver_mean.npy")
        receiver_std  = np.load("checkpoints/receiver_std.npy")

        receiver_meta = (receiver_raw - receiver_mean) / receiver_std
    else:
        receiver_meta = np.zeros((1, 6), dtype=np.float32)

    loc_pred = predict_location(
        loc_model,
        waveform,
        receiver_meta
    )[0]
    loc_mean = np.load("checkpoints/location_mean.npy")
    loc_std  = np.load("checkpoints/location_std.npy")

    delta_loc = loc_pred * loc_std + loc_mean

    lat = metadata["receiver_latitude"] + delta_loc[0]
    lon = metadata["receiver_longitude"] + delta_loc[1]
    depth = delta_loc[2]
    
    print(f"[Predict] Location   : lat={lat:.4f}°  lon={lon:.4f}°  depth={depth:.1f} km")
 
    report = run_alert_pipeline(
        detection_prob=det_prob,
        magnitude=magnitude,
        latitude=lat,
        longitude=lon,
        depth_km=depth,
        trace_name=trace_name,
        save_json=os.path.join(save_dir, "alert_report.json"),
    )
    plot_waveform(
        waveform[0],
        title=f"Waveform: {trace_name}",
        save_path=os.path.join(save_dir, "waveform.png"),
    )
 
    create_folium_alert_map(
        lat=lat, lon=lon,
        magnitude=magnitude,
        depth_km=depth,
        alert_info=report.to_dict(),
        save_path=os.path.join(save_dir, "alert_map.html"),
    )
 
    result = {
        "trace_name":    trace_name,
        "is_earthquake": is_eq,
        "detection_prob": det_prob,
        "magnitude": float(magnitude),
        "latitude": float(lat),
        "longitude": float(lon),
        "depth_km": float(depth),
        "risk_level":    report.level,
        "alert_active":  report.is_alert_active,
        "impact_radii":  report.impact_radii_km,
        "actions":       report.actions,
        "timestamp":     report.timestamp,
    }
 
    pred_path = os.path.join(save_dir, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Predict] Predictions saved → {pred_path}")
 
    return result
  
def main():
    parser = argparse.ArgumentParser(description="EWS Inference Pipeline")
    parser.add_argument("--trace_name", type=str, default=None)
    parser.add_argument("--chunk",      type=str, default="chunk2",
                        choices=["chunk1", "chunk2", "chunk4"])
    parser.add_argument("--demo",       action="store_true",
                        help="Run with synthetic waveform (no dataset needed)")
    parser.add_argument("--det_model",  type=str, default=None)
    parser.add_argument("--mag_model",  type=str, default=None)
    parser.add_argument("--loc_model",  type=str, default=None)
    parser.add_argument("--save_dir",   type=str, default="outputs")
    args = parser.parse_args()
 
    if args.demo:
        np.random.seed(42)
        t          = np.linspace(0, 60, 6000)
        waveform   = np.zeros((6000, 3), dtype=np.float32)
        p_onset    = 1500
        for ch in range(3):
            noise = np.random.randn(6000).astype(np.float32) * 0.05
            signal = np.zeros(6000, dtype=np.float32)
            signal[p_onset:] = (
                np.sin(2 * np.pi * 5 * t[p_onset:]) *
                np.exp(-0.03 * (t[p_onset:] - t[p_onset]))
            ).astype(np.float32) * (ch + 1)
            waveform[:, ch] = noise + signal
 
        waveform = waveform[np.newaxis, ...]  # (1, 6000, 3)
        trace_name = "synthetic_demo"
    else:
        if args.trace_name is None:
            print("Error: --trace_name is required unless --demo is set.")
            sys.exit(1)
 
        hdf5_map = {
            "chunk1": "data/raw/noise/chunk1/chunk1.hdf5",
            "chunk2": "data/raw/stead/chunk2/chunk2.hdf5",
            "chunk4": "data/raw/stead/chunk4/chunk4.hdf5",
        }
        from data.data_loader import load_single_trace
        waveform   = load_single_trace(args.trace_name, hdf5_map[args.chunk])
        trace_name = args.trace_name
        csv_map = {
        "chunk2": "data/raw/stead/chunk2/chunk2.csv",
        "chunk4": "data/raw/stead/chunk4/chunk4.csv",
    }

    meta_df = pd.read_csv(csv_map[args.chunk], low_memory=False)
    row = meta_df[meta_df["trace_name"] == trace_name].iloc[0]
 
    run_inference(
        waveform=waveform,
        trace_name=trace_name,
        det_path=args.det_model,
        mag_path=args.mag_model,
        loc_path=args.loc_model,
        save_dir=args.save_dir,
        metadata=row,
    )
 
 
if __name__ == "__main__":
    main()