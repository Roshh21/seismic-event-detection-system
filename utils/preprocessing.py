import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from typing import Optional, Tuple
  
def bandpass_filter(
    waveform: np.ndarray,
    lowcut:   float = 1.0,
    highcut:  float = 45.0,
    fs:       float = 100.0,
    order:    int   = 4,
) -> np.ndarray:
    nyq  = 0.5 * fs
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
 
    filtered = waveform.copy()
    for ch in range(waveform.shape[1]):
        filtered[:, ch] = filtfilt(b, a, waveform[:, ch])
    return filtered.astype(np.float32)
 
 
def notch_filter(
    waveform: np.ndarray,
    freq:     float = 60.0,
    quality:  float = 30.0,
    fs:       float = 100.0,
) -> np.ndarray:
    b, a    = iirnotch(freq / (0.5 * fs), quality)
    filtered = waveform.copy()
    for ch in range(waveform.shape[1]):
        filtered[:, ch] = filtfilt(b, a, waveform[:, ch])
    return filtered.astype(np.float32)

def taper(
    waveform:  np.ndarray,
    taper_pct: float = 0.05,
) -> np.ndarray:

    T   = waveform.shape[0]
    n   = int(T * taper_pct)
    win = np.hanning(2 * n)
 
    tapered = waveform.copy()
    for ch in range(waveform.shape[1]):
        tapered[:n,  ch] *= win[:n]
        tapered[-n:, ch] *= win[n:]
    return tapered.astype(np.float32)
 
def zscore_normalize(waveform: np.ndarray) -> np.ndarray:
    out = waveform.copy()
    for ch in range(waveform.shape[1]):
        mu  = out[:, ch].mean()
        std = out[:, ch].std()
        out[:, ch] = (out[:, ch] - mu) / (std + 1e-8)
    return out.astype(np.float32)
 
 
def peak_normalize(waveform: np.ndarray) -> np.ndarray:
    peak = np.abs(waveform).max()
    if peak < 1e-8:
        return waveform
    return (waveform / peak).astype(np.float32)
 
 
def rms_normalize(waveform: np.ndarray) -> np.ndarray:
    out = waveform.copy()
    for ch in range(waveform.shape[1]):
        rms = np.sqrt(np.mean(out[:, ch] ** 2))
        out[:, ch] /= (rms + 1e-8)
    return out.astype(np.float32)
 
 

def extract_p_window(
    waveform:       np.ndarray,
    p_arrival:      int,
    pre_p:          int = 100,
    window_length:  int = 600,
) -> np.ndarray:
    T   = waveform.shape[0]
    start = max(0, p_arrival - pre_p)
    end   = start + window_length
 
    if end > T:
        end   = T
        start = max(0, end - window_length)
 
    segment = waveform[start:end, :]
 
    if segment.shape[0] < window_length:
        pad = window_length - segment.shape[0]
        segment = np.pad(segment, ((0, pad), (0, 0)), mode="constant")
 
    return segment.astype(np.float32)
 
 
def extract_s_window(
    waveform:       np.ndarray,
    s_arrival:      int,
    pre_s:          int = 50,
    window_length:  int = 400,
) -> np.ndarray:
    return extract_p_window(waveform, s_arrival, pre_s, window_length)
 
 
def compute_snr(
    waveform:  np.ndarray,
    p_arrival: int,
    pre_window: int = 200,
) -> np.ndarray:
    C   = waveform.shape[1]
    snr = np.zeros(C, dtype=np.float32)
    for ch in range(C):
        noise_slice  = waveform[max(0, p_arrival - pre_window):p_arrival, ch]
        signal_slice = waveform[p_arrival:p_arrival + pre_window, ch]
 
        noise_rms  = np.sqrt(np.mean(noise_slice  ** 2) + 1e-12)
        signal_rms = np.sqrt(np.mean(signal_slice ** 2) + 1e-12)
        snr[ch]    = 20.0 * np.log10(signal_rms / noise_rms)
    return snr
 
 
def compute_peak_amplitude(waveform: np.ndarray) -> np.ndarray:
    return np.abs(waveform).max(axis=0).astype(np.float32)
 
 
def preprocess_waveform(
    waveform:   np.ndarray,
    apply_bp:   bool  = True,
    apply_taper: bool = True,
    norm:       str   = "zscore",  
    fs:         float = 100.0,
) -> np.ndarray:
    w = waveform.copy()
 
    if apply_taper:
        w = taper(w)
 
    if apply_bp:
        w = bandpass_filter(w, fs=fs)
 
    if norm == "zscore":
        w = zscore_normalize(w)
    elif norm == "peak":
        w = peak_normalize(w)
    elif norm == "rms":
        w = rms_normalize(w)
 
    return w
 
 
def batch_preprocess(
    batch: np.ndarray,
    **kwargs,
) -> np.ndarray:
    return np.stack(
        [preprocess_waveform(batch[i], **kwargs) for i in range(batch.shape[0])],
        axis=0,
    )
 
 
if __name__ == "__main__":
    dummy = np.random.randn(6000, 3).astype(np.float32)
    out   = preprocess_waveform(dummy)
    print(f"preprocess_waveform output shape: {out.shape}")
    print(f"Channel means (should be ~0): {out.mean(axis=0).round(4)}")
    print(f"Channel stds  (should be ~1): {out.std(axis=0).round(4)}")
    print("Preprocessing self-test passed ✓")