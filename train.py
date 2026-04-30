import argparse
import sys
import os
 
sys.path.insert(0, os.path.dirname(__file__))
 
from data.data_loader import (
    build_detection_dataset,
    build_magnitude_dataset,
    build_location_dataset,
)
from models.detection_model  import train_detection_model,  EarthquakeDetector
from models.magnitude_model  import train_magnitude_model,  MagnitudePredictor
from models.location_model   import train_location_model,   LocationPredictor
from utils.visualization     import plot_training_history
from utils.metrics           import (
    detection_metrics, print_detection_metrics,
    magnitude_metrics, print_magnitude_metrics,
    location_metrics,  print_location_metrics,
)
from models.detection_model  import predict_detection
from models.magnitude_model  import predict_magnitude
from models.location_model   import predict_location
 
  
def train_detection(args):
    print("\n" + "═" * 60)
    print("  PHASE 2 — Detection Model Training")
    print("═" * 60)
 
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = build_detection_dataset()
 
    model, history = train_detection_model(
        X_tr, y_tr, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
 
    plot_training_history(history, model_name="Detection Model")
 
    probs   = predict_detection(model, X_te)
    metrics = detection_metrics(y_te, probs)
    print_detection_metrics(metrics)
 
    print("[Detection] Training complete ✓")
    return model
 
 
def train_magnitude(args):
    print("\n" + "═" * 60)
    print("  PHASE 3 — Magnitude Prediction Model Training")
    print("═" * 60)

    # UPDATED: includes PSD features
    (X_tr, psd_tr, y_tr), (X_val, psd_val, y_val), (X_te, psd_te, y_te) = build_magnitude_dataset()

    model, history = train_magnitude_model(
        X_tr, psd_tr, y_tr,
        X_val, psd_val, y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )

    plot_training_history(history, model_name="Magnitude Model")

    preds = predict_magnitude(model, X_te, psd_te)

    metrics = magnitude_metrics(y_te, preds)
    print_magnitude_metrics(metrics)

    print("[Magnitude] Training complete ✓")
    return model
 
 
def train_location(args):
    print("\n" + "═" * 60)
    print("  PHASE 4 — Location Prediction Model Training")
    print("═" * 60)
 
    (X_tr, meta_tr, y_tr), (X_val, meta_val, y_val), (X_te, meta_te, y_te) = build_location_dataset()
 
    model, history = train_location_model(
        X_tr, meta_tr, y_tr,
        X_val, meta_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
 
    plot_training_history(history, model_name="Location Model")
 
    preds = predict_location(model, X_te, meta_te)
    metrics = location_metrics(y_te, preds)
    print_location_metrics(metrics)
 
    print("[Location] Training complete ✓")
    return model
 
def main():
    parser = argparse.ArgumentParser(description="Train Earthquake EWS models")
    parser.add_argument(
        "--model", choices=["all", "detection", "magnitude", "location"],
        default="all", help="Which model to train (default: all)"
    )
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=5e-4)
    args = parser.parse_args()
 
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║     Earthquake Early Warning System — Training    ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  Target model : {args.model}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
 
    import os
    os.makedirs("outputs", exist_ok=True)
 
    if args.model in ("all", "detection"):
        train_detection(args)
 
    if args.model in ("all", "magnitude"):
        train_magnitude(args)
 
    if args.model in ("all", "location"):
        train_location(args)
 
    print("\n" + "═" * 60)
    print("    Training complete. Checkpoints saved to checkpoints/")
    print("    Plots saved to outputs/")
    print("═" * 60 + "\n")
 
 
if __name__ == "__main__":
    main()