import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data.data_loader import (
    build_detection_dataset,
    build_magnitude_dataset,
    build_location_dataset,
)

from models.detection_model import (
    load_detection_model,
    predict_detection,
)

from models.magnitude_model import (
    load_magnitude_model,
    predict_magnitude,
)

from models.location_model import (
    load_location_model,
    predict_location,
)

from utils.metrics import (
    detection_metrics,
    print_detection_metrics,
    magnitude_metrics,
    print_magnitude_metrics,
    location_metrics,
    print_location_metrics,
)

from utils.visualization import (
    plot_confusion_matrix,
    plot_magnitude_scatter,
    plot_location_predictions,
)

def evaluate_detection(batch_size: int) -> dict:
    print("\n── Evaluating Detection Model ")

    *_, (X_te, y_te) = build_detection_dataset()

    model = load_detection_model()

    probs = predict_detection(
        model,
        X_te,
        batch_size=batch_size,
    )

    metrics = detection_metrics(y_te, probs)

    print_detection_metrics(metrics)

    plot_confusion_matrix(
        metrics["confusion_matrix"]
    )

    return metrics


def evaluate_magnitude(batch_size: int) -> dict:
    print("\n── Evaluating Magnitude Model")

    *_, (X_te, psd_te, y_te) = build_magnitude_dataset()

    model = load_magnitude_model()

    preds = predict_magnitude(
        model,
        X_te,
        psd_te,
        batch_size=batch_size,
    )

    mag_mean = np.load("checkpoints/magnitude_mean.npy")
    mag_std  = np.load("checkpoints/magnitude_std.npy")

    y_true = y_te * mag_std + mag_mean
    y_pred = preds * mag_std + mag_mean

    metrics = magnitude_metrics(y_true, y_pred)

    print_magnitude_metrics(metrics)

    plot_magnitude_scatter(
        y_true,
        y_pred,
    )

    return metrics

def evaluate_location(batch_size: int) -> dict:
    print("\n── Evaluating Location Model ")
    *_, (X_te, meta_te, y_te) = build_location_dataset()

    model = load_location_model()

    preds = predict_location(
        model,
        X_te,
        meta_te,
        batch_size=batch_size,
    )
    metrics = location_metrics(y_te, preds)
    print_location_metrics(metrics)
    loc_mean = np.load("checkpoints/location_mean.npy")
    loc_std = np.load("checkpoints/location_std.npy")

    receiver_mean = np.load("checkpoints/receiver_mean.npy")
    receiver_std = np.load("checkpoints/receiver_std.npy")

    receiver_real = (
        meta_te[:, :3] * receiver_std[:3]
        + receiver_mean[:3]
    )

    true_delta = y_te * loc_std + loc_mean
    pred_delta = preds * loc_std + loc_mean

    y_true = np.zeros_like(true_delta)
    y_pred = np.zeros_like(pred_delta)

    y_true[:, 0] = receiver_real[:, 0] + true_delta[:, 0]
    y_pred[:, 0] = receiver_real[:, 0] + pred_delta[:, 0]

    y_true[:, 1] = receiver_real[:, 1] + true_delta[:, 1]
    y_pred[:, 1] = receiver_real[:, 1] + pred_delta[:, 1]

    y_true[:, 2] = true_delta[:, 2]
    y_pred[:, 2] = pred_delta[:, 2]

    max_points = 3000

    if len(y_true) > max_points:
        idx = np.random.choice(
            len(y_true),
            max_points,
            replace=False,
        )

        y_true_plot = y_true[idx]
        y_pred_plot = y_pred[idx]

    else:
        y_true_plot = y_true
        y_pred_plot = y_pred

    plot_location_predictions(
        y_true_plot,
        y_pred_plot,
    )

    return metrics

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Earthquake EWS models"
    )

    parser.add_argument(
        "--model",
        choices=[
            "all",
            "detection",
            "magnitude",
            "location",
        ],
        default="all",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    print("\n╔" + "═" * 58 + "╗")
    print("║      Earthquake EWS — Model Evaluation             ║")
    print("╚" + "═" * 58 + "╝")

    if args.model in ("all", "detection"):
        evaluate_detection(args.batch_size)

    if args.model in ("all", "magnitude"):
        evaluate_magnitude(args.batch_size)

    if args.model in ("all", "location"):
        evaluate_location(args.batch_size)

    print("\n  Evaluation complete. Plots saved to outputs/\n")


if __name__ == "__main__":
    main()