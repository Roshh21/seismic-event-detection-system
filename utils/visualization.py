import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
 
try:
    import folium
    from folium.plugins import HeatMap
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
 
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
 
DARK_BG  = "#0d1117"
ACCENT   = "#00d4aa"
WARN     = "#f5a623"
DANGER   = "#e74c3c"
GRID_CLR = "#1f2937"
 
plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    DARK_BG,
    "axes.edgecolor":    GRID_CLR,
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        GRID_CLR,
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
})
 
CHANNEL_COLORS = ["#58a6ff", "#3fb950", "#f78166"]
CHANNEL_NAMES  = ["East (E)", "North (N)", "Vertical (Z)"]
 
  
def plot_waveform(
    waveform:   np.ndarray,
    fs:         float = 100.0,
    title:      str   = "Seismic Waveform",
    p_arrival:  int   = None,
    s_arrival:  int   = None,
    save_path:  str   = None,
) -> None:
    T   = waveform.shape[0]
    t   = np.arange(T) / fs
 
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(title, fontsize=14, color=ACCENT, fontweight="bold")
 
    for ch, ax in enumerate(axes):
        ax.plot(t, waveform[:, ch], color=CHANNEL_COLORS[ch], linewidth=0.7, alpha=0.9)
        ax.set_ylabel(CHANNEL_NAMES[ch], fontsize=9)
        ax.grid(True, alpha=0.4)
 
        if p_arrival is not None:
            ax.axvline(p_arrival / fs, color="#f5a623", linewidth=1.2,
                       linestyle="--", label="P" if ch == 0 else "")
        if s_arrival is not None:
            ax.axvline(s_arrival / fs, color="#e74c3c", linewidth=1.2,
                       linestyle="--", label="S" if ch == 0 else "")
 
        if ch == 0 and (p_arrival or s_arrival):
            ax.legend(fontsize=8, loc="upper right",
                      facecolor=DARK_BG, edgecolor=GRID_CLR)
 
    axes[-1].set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout()
 
    path = save_path or str(OUTPUT_DIR / "waveform.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[Viz] Waveform saved → {path}")
 
  
def plot_training_history(
    history:    dict,
    model_name: str = "Model",
    save_path:  str = None,
) -> None:
    has_acc = "accuracy" in history
 
    fig, axes = plt.subplots(1, 2 if has_acc else 1,
                              figsize=(14 if has_acc else 7, 5))
    if not has_acc:
        axes = [axes]
 
    fig.suptitle(f"{model_name} – Training History",
                 fontsize=13, color=ACCENT, fontweight="bold")
 
    ax = axes[0]
    ax.plot(history["loss"],     color=CHANNEL_COLORS[0], label="Train Loss", linewidth=1.5)
    ax.plot(history["val_loss"], color=WARN,              label="Val Loss",   linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss", color="#c9d1d9")
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_CLR)
    ax.grid(True, alpha=0.3)
 
    if has_acc:
        ax = axes[1]
        ax.plot(history["accuracy"],     color=CHANNEL_COLORS[1], label="Train Acc", linewidth=1.5)
        ax.plot(history["val_accuracy"], color=WARN,              label="Val Acc",   linewidth=1.5, linestyle="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy", color="#c9d1d9")
        ax.legend(facecolor=DARK_BG, edgecolor=GRID_CLR)
        ax.grid(True, alpha=0.3)
 
    plt.tight_layout()
    path = save_path or str(OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[Viz] Training history saved → {path}")
 
 
def plot_magnitude_scatter(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    save_path: str = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle("Magnitude Prediction", fontsize=13, color=ACCENT, fontweight="bold")
 
    ax.scatter(y_true, y_pred, alpha=0.4, s=15,
               c=CHANNEL_COLORS[0], edgecolors="none")
 
    mn = min(y_true.min(), y_pred.min()) - 0.3
    mx = max(y_true.max(), y_pred.max()) + 0.3
    ax.plot([mn, mx], [mn, mx], color=WARN, linewidth=1.5, linestyle="--", label="Perfect")
 
    ax.set_xlabel("True Magnitude");  ax.set_ylabel("Predicted Magnitude")
    ax.set_xlim(mn, mx);             ax.set_ylim(mn, mx)
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_CLR)
    ax.grid(True, alpha=0.3)
 
    path = save_path or str(OUTPUT_DIR / "magnitude_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[Viz] Magnitude scatter saved → {path}")
 
 
def plot_confusion_matrix(
    cm:        np.ndarray,
    labels:    list = ["Noise", "Earthquake"],
    save_path: str  = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle("Confusion Matrix", fontsize=13, color=ACCENT, fontweight="bold")
 
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar=True, linewidths=0.5, linecolor=DARK_BG,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
 
    path = save_path or str(OUTPUT_DIR / "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[Viz] Confusion matrix saved → {path}")
 
 
 
def plot_location_predictions(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    n_samples: int = 300,
    save_path: str = None,
) -> None:
    idx = np.random.choice(len(y_true), min(n_samples, len(y_true)), replace=False)
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle("Epicenter Prediction (lat/lon)", fontsize=13,
                 color=ACCENT, fontweight="bold")
 
    ax.scatter(y_true[idx, 1], y_true[idx, 0], s=12,
               c=CHANNEL_COLORS[1], alpha=0.6, label="True", zorder=3)
    ax.scatter(y_pred[idx, 1], y_pred[idx, 0], s=12, marker="x",
               c=WARN, alpha=0.6, label="Predicted", zorder=3)
 
    for i in idx[:50]:   
        ax.plot([y_true[i, 1], y_pred[i, 1]],
                [y_true[i, 0], y_pred[i, 0]],
                color="#4a5568", linewidth=0.4, alpha=0.5)
 
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180);     ax.set_ylim(-90, 90)
    ax.axhline(0, color=GRID_CLR, linewidth=0.5)
    ax.axvline(0, color=GRID_CLR, linewidth=0.5)
    ax.legend(facecolor=DARK_BG, edgecolor=GRID_CLR, fontsize=9)
    ax.grid(True, alpha=0.3)
 
    path = save_path or str(OUTPUT_DIR / "location_prediction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[Viz] Location prediction saved → {path}")
 
  
def create_folium_alert_map(
    lat:        float,
    lon:        float,
    magnitude:  float,
    depth_km:   float,
    alert_info: dict,
    save_path:  str = None,
) -> None:
    if not FOLIUM_AVAILABLE:
        print("[Viz] Folium not installed – skipping map generation.")
        return
 
    m = folium.Map(location=[lat, lon], zoom_start=6, tiles="CartoDB dark_matter")
 
    level   = alert_info.get("level", "LOW")
    color   = {"LOW": "green", "MODERATE": "orange", "HIGH": "red"}.get(level, "blue")
 
    folium.CircleMarker(
        location=[lat, lon],
        radius=10,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.8,
        tooltip=folium.Tooltip(
            f"<b>Epicenter</b><br>"
            f"Magnitude: {magnitude:.2f}<br>"
            f"Depth: {depth_km:.1f} km<br>"
            f"Risk: {level}",
        ),
    ).add_to(m)
 
    radii = alert_info.get("impact_radii_km", {})
    styles = {
        "high":     ("#e74c3c", 0.35),
        "moderate": ("#f5a623", 0.20),
        "low":      ("#2ecc71", 0.10),
    }
    for zone, r_km in radii.items():
        clr, opacity = styles.get(zone, ("#aaa", 0.1))
        folium.Circle(
            location=[lat, lon],
            radius=r_km * 1000,          
            color=clr,
            fill=True,
            fill_color=clr,
            fill_opacity=opacity,
            tooltip=f"{zone.title()} zone ({r_km:.0f} km)",
        ).add_to(m)
 
    folium.Marker(
        location=[lat + 0.5, lon],
        icon=folium.DivIcon(
            html=f"""
            <div style="
                background:{color};color:white;
                padding:6px 12px;border-radius:4px;
                font-family:monospace;font-size:13px;
                font-weight:bold;white-space:nowrap;
                box-shadow:0 2px 8px rgba(0,0,0,0.6);
            ">
            ⚠ M{magnitude:.1f} · {level} RISK
            </div>""",
        ),
    ).add_to(m)
 
    path = save_path or str(OUTPUT_DIR / "alert_map.html")
    m.save(path)
    print(f"[Viz] Alert map saved → {path}")
 
 
if __name__ == "__main__":
    dummy_waveform = np.random.randn(6000, 3).astype(np.float32)
    plot_waveform(dummy_waveform, title="Smoke-test Waveform")
 
    history = {
        "loss":         list(np.linspace(1.0, 0.2, 30) + np.random.randn(30) * 0.03),
        "val_loss":     list(np.linspace(1.1, 0.3, 30) + np.random.randn(30) * 0.03),
        "accuracy":     list(np.linspace(0.6, 0.97, 30)),
        "val_accuracy": list(np.linspace(0.55, 0.94, 30)),
    }
    plot_training_history(history, model_name="Detection Model")
    print("Visualization self-test passed ✓")