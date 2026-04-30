# Earthquake Early Warning System (EWS)

A deep learning-based system for:

* **Earthquake Detection** (Earthquake vs Noise)
* **Magnitude Estimation**
* **Location Prediction** (Latitude, Longitude, Depth)
* **Risk Alert Generation**
* **Dashboard Visualization**

This project uses seismic waveform data from the **STEAD dataset** and implements hybrid **CNN + Transformer architectures** for earthquake detection, seismic analysis, and event prediction.

---

## Overview

The system is designed as a complete earthquake monitoring pipeline:

### Detection Model

* Classifies seismic signals as:

  * Earthquake
  * Noise
* Uses waveform data only

### Magnitude Model

* Predicts earthquake magnitude
* Uses:

  * Waveform data
  * **PSD (Power Spectral Density) features**
* PSD improves frequency-domain analysis and strengthens magnitude prediction performance

### Location Model

* Predicts:

  * Latitude
  * Longitude
  * Depth
* Uses:

  * Waveform data
  * Station metadata
  * Arrival information

### Alert Engine

* Generates earthquake risk levels:

  * Low
  * Moderate
  * High

### Dashboard

* Interactive visualization of:

  * Detection results
  * Magnitude
  * Epicenter location
  * Alert levels

---

# Project Structure

```bash
EQ-DETECTION-1/
│
├── alerts/
│   └── alert_engine.py
│
├── checkpoints/
│
├── dashboard/
│   └── dashboard.html
│
├── data/
│   ├── raw/
│   │   ├── noise/
│   │   └── stead/
│   │       ├── chunk2/
│   │       └── chunk4/
│   │
│   └── data_loader.py
│
├── models/
│   ├── detection_model.py
│   ├── magnitude_model.py
│   └── location_model.py
│
├── outputs/
├── outputs_indian_test/
│
├── utils/
│   ├── metrics.py
│   ├── preprocessing.py
│   └── visualization.py
│
├── debug_location_scale.py
├── evaluate.py
├── predict.py
├── test.py
├── train.py
│
├── requirements.txt
└── README.md
```

---

# Model Overview

| Model     | Architecture                     | Input               | Output                       | Task                 |
| --------- | -------------------------------- | ------------------- | ---------------------------- | -------------------- |
| Detection | CNN + Transformer                | Waveform            | Binary                       | Earthquake / Noise   |
| Magnitude | CNN + Transformer + PSD          | Waveform + PSD      | Scalar                       | Magnitude Prediction |
| Location  | CNN + Residual CNN + Transformer | Waveform + Metadata | Latitude / Longitude / Depth | Epicenter Prediction |

---

## PSD in Magnitude Prediction

Power Spectral Density (PSD) extracts frequency-based energy information from seismic signals.

### Purpose:

* Improves strength estimation
* Captures energy distribution
* Enhances regression performance

### Summary:

* Waveform → Signal shape
* PSD → Signal energy

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/Roshh21/EQ-DETECTION.git
cd EQ-DETECTION
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Setup

This project uses the **STEAD (Stanford Earthquake Dataset)**.

## Required Dataset Chunks:

* **Chunk 1** → Noise / Non-earthquake signals
* **Chunk 2** → Earthquake waveform data
* **Chunk 4** → Additional earthquake waveform data

## Download Links:

* Official Repository:
  https://github.com/smousavi05/STEAD

* Zenodo Dataset Download:
  https://zenodo.org/record/3894660

---

## Organize Dataset as:

```bash
data/
└── raw/
    ├── noise/          # Chunk 1
    └── stead/
        ├── chunk2/     # Chunk 2
        └── chunk4/     # Chunk 4
```

---

## Update Dataset Paths

Inside:

```bash
data/data_loader.py
```

Example:

```python
NOISE_PATH = "data/raw/noise/"
STEAD_CHUNK2_PATH = "data/raw/stead/chunk2/"
STEAD_CHUNK4_PATH = "data/raw/stead/chunk4/"
```

---

# Execution Steps

## Train All Models

```bash
python train.py
```

---

## Evaluate All Models

```bash
python evaluate.py
```

---

## Run Prediction

```bash
python predict.py
```
---

## Run Prediction on Specific Seismic Trace

Use the following command format:

```bash
python predict.py --chunk <chunk_name> --trace_name "<trace_name>"
```
---
## Get Random Trace Names for Testing

To retrieve valid random seismic trace names from available dataset chunks (chunk2 or chunk4) for prediction testing:

```bash
python test.py
```

---

# Dashboard

Open:

```bash
dashboard/dashboard.html
```

### Dashboard Includes:

* Detection results
* Magnitude estimation
* Location prediction
* Alert level

---

# Alert Levels

| Magnitude | Risk Level |
| --------- | ---------- |
| < 4.0     | Low        |
| 4.0 – 6.0 | Moderate   |
| > 6.0     | High       |

---

# Dataset Details

## STEAD Dataset Includes:

* Seismic waveform traces
* Event metadata
* Station metadata
* Magnitude labels
* Source coordinates
* Arrival picks

## Dataset Usage:

* Chunk 1 → Detection negative samples
* Chunk 2 → Detection, Magnitude, Location
* Chunk 4 → Additional earthquake training/testing

---

# Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* CNN
* Transformer Networks
* HTML/CSS
* Leaflet.js

---
