import numpy as np
from pathlib import Path

from predict import run_inference


def normalize_waveform(waveform):
    """
    Standard per-channel z-score normalization
    """
    mean = waveform.mean(axis=0, keepdims=True)
    std = waveform.std(axis=0, keepdims=True) + 1e-6
    return (waveform - mean) / std


BASE_PATH = Path("data/test1")
TRACE_BASE = "GUA_20091116_041239"

EW_FILE = BASE_PATH / f"{TRACE_BASE}.ew"
NS_FILE = BASE_PATH / f"{TRACE_BASE}.ns"
VT_FILE = BASE_PATH / f"{TRACE_BASE}.vt"

def load_component(filepath):
    values = []

    with open(filepath, "r") as f:
        for line in f:
            try:
                values.append(float(line.strip()))
            except ValueError:
                continue  

    return np.array(values, dtype=np.float32)

def build_waveform():
    print("[Indian Test] Loading waveform components...")

    ew = load_component(EW_FILE)
    ns = load_component(NS_FILE)
    vt = load_component(VT_FILE)

    print(f"EW samples: {len(ew)}")
    print(f"NS samples: {len(ns)}")
    print(f"VT samples: {len(vt)}")

    min_len = min(len(ew), len(ns), len(vt))
    ew, ns, vt = ew[:min_len], ns[:min_len], vt[:min_len]

    waveform = np.stack([ew, ns, vt], axis=1)
    waveform = waveform[::2]
    target_len = 6000

    if len(waveform) > target_len:
        waveform = waveform[:target_len]
    elif len(waveform) < target_len:
        pad = np.zeros((target_len - len(waveform), 3), dtype=np.float32)
        waveform = np.vstack([waveform, pad])

    print(f"Final waveform shape: {waveform.shape}")

    waveform = normalize_waveform(waveform)

    return waveform

def build_indian_metadata():
    """
    Approximate metadata for Indian station.
    Since exact P/S picks may not be available,
    use rough placeholders.
    """

    receiver_lat = 29.866
    receiver_lon = 77.901
    receiver_elev = 262.0

    p_arrival = 1000.0
    s_arrival = 2500.0
    ps_gap = s_arrival - p_arrival

    metadata = {
        "receiver_latitude": receiver_lat,
        "receiver_longitude": receiver_lon,
        "receiver_elevation_m": receiver_elev,
        "p_arrival_sample": p_arrival,
        "s_arrival_sample": s_arrival,
    }

    return metadata

def main():
    waveform = build_waveform()

    waveform = waveform[np.newaxis, ...]

    metadata = build_indian_metadata()

    run_inference(
        waveform=waveform,
        trace_name=TRACE_BASE,
        metadata=metadata,
        save_dir="outputs_indian_test"
    )


if __name__ == "__main__":
    main()