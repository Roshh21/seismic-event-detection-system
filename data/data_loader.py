import os
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.signal import welch

CHUNK1_PATH = "data/raw/noise/chunk1/chunk1.hdf5"   
CHUNK1_CSV  = "data/raw/noise/chunk1/chunk1.csv"    

CHUNK2_PATH = "data/raw/stead/chunk2/chunk2.hdf5"   
CHUNK2_CSV  = "data/raw/stead/chunk2/chunk2.csv"

CHUNK4_PATH = "data/raw/stead/chunk4/chunk4.hdf5"
CHUNK4_CSV  = "data/raw/stead/chunk4/chunk4.csv"   

WAVEFORM_LENGTH   = 6000          
NUM_CHANNELS      = 3             
RANDOM_SEED       = 42
MAX_NOISE_SAMPLES = 100000      
MAX_EQ_SAMPLES    = 100000     


def load_waveform(hdf5_file: h5py.File, trace_name: str) -> np.ndarray:
    dataset = hdf5_file.get(f"data/{trace_name}")
    if dataset is None:
        raise KeyError(f"Trace '{trace_name}' not found in HDF5 file.")
    waveform = np.array(dataset)          


    if waveform.shape[0] < WAVEFORM_LENGTH:
        pad = WAVEFORM_LENGTH - waveform.shape[0]
        waveform = np.pad(waveform, ((0, pad), (0, 0)), mode="constant")
    else:
        waveform = waveform[:WAVEFORM_LENGTH, :]

    return waveform.astype(np.float32)


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    normalized = waveform.copy()
    for ch in range(waveform.shape[1]):
        mu  = normalized[:, ch].mean()
        std = normalized[:, ch].std()
        if std > 1e-8:
            normalized[:, ch] = (normalized[:, ch] - mu) / std
        else:
            normalized[:, ch] = 0.0
    return normalized


def load_noise_waveforms(
    hdf5_path: str = CHUNK1_PATH,
    csv_path:  str = CHUNK1_CSV,
    max_samples: int = MAX_NOISE_SAMPLES,
) -> np.ndarray:
    print(f"[DataLoader] Loading noise waveforms from {hdf5_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.sample(min(len(df), max_samples), random_state=RANDOM_SEED).reset_index(drop=True)

    waveforms = []
    with h5py.File(hdf5_path, "r") as f:
        for trace_name in tqdm(df["trace_name"], desc="Noise"):
            try:
                w = load_waveform(f, trace_name)
                w = normalize_waveform(w)
                waveforms.append(w)
            except Exception as e:
                pass  
    X = np.stack(waveforms, axis=0)
    print(f"[DataLoader] Noise loaded: {X.shape}")
    return X


def load_earthquake_waveforms(
    hdf5_path: str = CHUNK2_PATH,
    csv_path: str = CHUNK2_CSV,
    max_samples: int = MAX_EQ_SAMPLES,
) -> tuple:
    print(f"[DataLoader] Loading earthquake waveforms from {hdf5_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)

    required = [
    "trace_name",
    "source_magnitude",
    "source_latitude",
    "source_longitude",
    "source_depth_km",
    "receiver_latitude",
    "receiver_longitude",
    "receiver_elevation_m",
    "p_arrival_sample",
    "s_arrival_sample",
    ]

    df = df.dropna(subset=required)

    extreme_mag = df[df["source_magnitude"] >= 7.0]
    high_mag    = df[
        (df["source_magnitude"] >= 5.5) &
        (df["source_magnitude"] < 7.0)
    ]
    mid_mag     = df[
        (df["source_magnitude"] >= 4.0) &
        (df["source_magnitude"] < 5.5)
    ]

    base_df = df.sample(
        min(len(df), max_samples),
        random_state=RANDOM_SEED,
    )

    extreme_boost = extreme_mag.sample(
        n=min(max(len(extreme_mag) * 15, 5000), max_samples // 2),
        replace=True,
        random_state=RANDOM_SEED,
    ) if len(extreme_mag) > 0 else pd.DataFrame()

    high_boost = high_mag.sample(
        n=min(max(len(high_mag) * 8, 4000), max_samples // 3),
        replace=True,
        random_state=RANDOM_SEED,
    ) if len(high_mag) > 0 else pd.DataFrame()

    mid_boost = mid_mag.sample(
        n=min(max(len(mid_mag) * 3, 3000), max_samples // 4),
        replace=True,
        random_state=RANDOM_SEED,
    ) if len(mid_mag) > 0 else pd.DataFrame()

    df = pd.concat([
        base_df,
        extreme_boost,
        high_boost,
        mid_boost,
    ])

    df = df.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    df = df.iloc[:max_samples].reset_index(drop=True)

    waveforms = []
    valid_idx = []

    with h5py.File(hdf5_path, "r") as f:
        for i, trace_name in enumerate(
            tqdm(df["trace_name"], desc="Earthquake")
        ):
            try:
                w = load_waveform(f, trace_name)
                w = normalize_waveform(w)
                waveforms.append(w)
                valid_idx.append(i)
            except Exception:
                pass

    X = np.stack(waveforms, axis=0)
    meta = df.iloc[valid_idx][required].reset_index(drop=True)

    print(f"[DataLoader] Earthquake loaded: {X.shape}")
    return X, meta

def extract_psd_features(waveform: np.ndarray) -> np.ndarray:
    features = []

    for ch in range(waveform.shape[1]):
        freqs, psd = welch(
            waveform[:, ch],
            fs=100,
            nperseg=256
        )

        total_power = np.sum(psd)
        peak_freq = freqs[np.argmax(psd)]
        mean_power = np.mean(psd)

        low_band = np.sum(psd[(freqs >= 0.5) & (freqs < 2)])
        mid_band = np.sum(psd[(freqs >= 2) & (freqs < 8)])
        high_band = np.sum(psd[(freqs >= 8)])

        features.extend([
            total_power,
            peak_freq,
            mean_power,
            low_band,
            mid_band,
            high_band,
        ])

    return np.array(features, dtype=np.float32)

def load_earthquake_waveforms_multi(
    max_samples: int = MAX_EQ_SAMPLES,
):
    datasets = [
        (CHUNK2_PATH, CHUNK2_CSV),
        (CHUNK4_PATH, CHUNK4_CSV),
    ]

    all_waves = []
    all_meta = []

    per_chunk = max_samples // len(datasets)

    for hdf5_path, csv_path in datasets:
        X_chunk, meta_chunk = load_earthquake_waveforms(
            hdf5_path=hdf5_path,
            csv_path=csv_path,
            max_samples=per_chunk,
        )

        all_waves.append(X_chunk)
        all_meta.append(meta_chunk)

    X = np.concatenate(all_waves, axis=0)
    meta = pd.concat(all_meta, ignore_index=True)

    idx = np.random.RandomState(RANDOM_SEED).permutation(len(X))

    X = X[idx]
    meta = meta.iloc[idx].reset_index(drop=True)

    print(f"[DataLoader] Combined earthquake loaded: {X.shape}")
    return X, meta


def build_detection_dataset(
    test_size: float = 0.15,
    val_size:  float = 0.15,
):
    X_eq = load_earthquake_waveforms_multi()[0]
    X_noise = load_noise_waveforms(max_samples=len(X_eq))

    n = min(len(X_eq), len(X_noise))
    X_eq    = X_eq[:n]
    X_noise = X_noise[:n]

    X = np.concatenate([X_eq, X_noise], axis=0)
    y = np.concatenate([np.ones(n), np.zeros(n)], axis=0).astype(np.float32)

    idx = np.random.RandomState(RANDOM_SEED).permutation(len(X))
    X, y = X[idx], y[idx]

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=test_size / (test_size + val_size),
        random_state=RANDOM_SEED,
    )
    print(f"[Detection] Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_magnitude_dataset(
    test_size: float = 0.15,
    val_size: float = 0.15,
):
    X, meta = load_earthquake_waveforms_multi()

    psd_features = np.stack(
        [extract_psd_features(w) for w in X],
        axis=0
    )

    psd_mean = psd_features.mean(axis=0)
    psd_std = psd_features.std(axis=0) + 1e-6

    np.save("checkpoints/psd_mean.npy", psd_mean)
    np.save("checkpoints/psd_std.npy", psd_std)

    psd_features = (psd_features - psd_mean) / psd_std

    y_raw = meta["source_magnitude"].values.astype(np.float32)

    mag_mean = y_raw.mean()
    mag_std = y_raw.std() + 1e-6

    np.save("checkpoints/magnitude_mean.npy", mag_mean)
    np.save("checkpoints/magnitude_std.npy", mag_std)

    y = (y_raw - mag_mean) / mag_std

    (
        X_train,
        X_tmp,
        psd_train,
        psd_tmp,
        y_train,
        y_tmp,
    ) = train_test_split(
        X,
        psd_features,
        y,
        test_size=test_size + val_size,
        random_state=RANDOM_SEED,
    )

    (
        X_val,
        X_test,
        psd_val,
        psd_test,
        y_val,
        y_test,
    ) = train_test_split(
        X_tmp,
        psd_tmp,
        y_tmp,
        test_size=test_size / (test_size + val_size),
        random_state=RANDOM_SEED,
    )

    print(
        f"[Magnitude] Train={len(X_train)}, "
        f"Val={len(X_val)}, Test={len(X_test)}"
    )

    return (
        (X_train, psd_train, y_train),
        (X_val, psd_val, y_val),
        (X_test, psd_test, y_test),
    )

def build_location_dataset(
    test_size: float = 0.15,
    val_size:  float = 0.15,
):
    X, meta = load_earthquake_waveforms_multi()
    source_lat = meta["source_latitude"].values.astype(np.float32)
    source_lon = meta["source_longitude"].values.astype(np.float32)
    source_dep = meta["source_depth_km"].values.astype(np.float32)

    recv_lat = meta["receiver_latitude"].values.astype(np.float32)
    recv_lon = meta["receiver_longitude"].values.astype(np.float32)

    delta_lat = source_lat - recv_lat
    delta_lon = source_lon - recv_lon

    source_labels = np.stack(
    [delta_lat, delta_lon, source_dep],
    axis=1
    )
    
    receiver_meta = meta[
    [
        "receiver_latitude",
        "receiver_longitude",
        "receiver_elevation_m",
        "p_arrival_sample",
        "s_arrival_sample",
    ]
    ].values.astype(np.float32)

    ps_gap = (
        receiver_meta[:, 4] - receiver_meta[:, 3]
    ).reshape(-1, 1)

    receiver_meta = np.concatenate(
        [receiver_meta, ps_gap],
        axis=1
    )

    receiver_mean = receiver_meta.mean(axis=0)
    receiver_std  = receiver_meta.std(axis=0) + 1e-6

    np.save("checkpoints/receiver_mean.npy", receiver_mean)
    np.save("checkpoints/receiver_std.npy", receiver_std)

    receiver_meta = (receiver_meta - receiver_mean) / receiver_std

    loc_mean = source_labels.mean(axis=0)
    loc_std  = source_labels.std(axis=0) + 1e-6

    np.save("checkpoints/location_mean.npy", loc_mean)
    np.save("checkpoints/location_std.npy", loc_std)

    y = (source_labels - loc_mean) / loc_std

    (
        X_train,
        X_tmp,
        meta_train,
        meta_tmp,
        y_train,
        y_tmp,
    ) = train_test_split(
        X,
        receiver_meta,
        y,
        test_size=test_size + val_size,
        random_state=RANDOM_SEED,
    )

    (
        X_val,
        X_test,
        meta_val,
        meta_test,
        y_val,
        y_test,
    ) = train_test_split(
        X_tmp,
        meta_tmp,
        y_tmp,
        test_size=test_size / (test_size + val_size),
        random_state=RANDOM_SEED,
    )

    print(
        f"[Location] Train={len(X_train)}, "
        f"Val={len(X_val)}, Test={len(X_test)}"
    )

    return (
        (X_train, meta_train, y_train),
        (X_val, meta_val, y_val),
        (X_test, meta_test, y_test),
    )


def load_single_trace(
    trace_name: str,
    hdf5_path: str,
) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as f:
        w = load_waveform(f, trace_name)
    w = normalize_waveform(w)
    return w[np.newaxis, ...]   


if __name__ == "__main__":
    print("DataLoader self-test skipped (configure paths first).")
    print("Required files:")
    print(f"  {CHUNK1_PATH}, {CHUNK1_CSV}")
    print(f"  {CHUNK2_PATH}, {CHUNK2_CSV}")