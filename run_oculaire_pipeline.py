#!/usr/bin/env python3
import os
import argparse
from src.preprocessing.harvard_npz_adapter import convert_npz_to_png
from src.training.trainer import train_model, evaluate_model
from src.utils.visualization import plot_results

def process_folder(npz_folder, out_folder, key_name):
    """Convert all .npz files in a folder to PNGs."""
    os.makedirs(out_folder, exist_ok=True)
    convert_npz_to_png(npz_folder, out_folder, key_name)

def main(npz=None, out=None):
    # ---------------------------
    # If single npz specified, just convert it
    # ---------------------------
    if npz and out:
        print(f"🔹 Converting single file: {npz} → {out}")
        convert_npz_to_png(npz_folder=os.path.dirname(npz),
                           out_folder=out,
                           key_name="bscans")  # or "rnflt" depending on file
    else:
        # ---------------------------
        # Full dataset mode
        # ---------------------------
        RAW_BSCAN = "data/raw/Bscan"
        RAW_RNFLT = "data/raw/RNFLT"
        PROCESSED_BSCAN = "data/processed/Bscan"
        PROCESSED_RNFLT = "data/processed/RNFLT"

        print("🔹 Converting Bscan dataset...")
        process_folder(RAW_BSCAN, PROCESSED_BSCAN, key_name="bscans")

        print("🔹 Converting RNFLT dataset...")
        process_folder(RAW_RNFLT, PROCESSED_RNFLT, key_name="rnflt")

        # ---------------------------
        # Train & Evaluate
        # ---------------------------
        print("🔹 Training model...")
        model = train_model(bscan_dir=PROCESSED_BSCAN, rnflt_dir=PROCESSED_RNFLT, output_model_dir="models/")

        print("🔹 Evaluating model...")
        metrics = evaluate_model(model, bscan_dir=PROCESSED_BSCAN, rnflt_dir=PROCESSED_RNFLT)

        print("🔹 Visualizing results...")
        plot_results(metrics)

    print("✅ Pipeline finished successfully!")

# ---------------------------
# CLI Arguments
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCULAIRE pipeline")
    parser.add_argument("--npz", type=str, help="Path to single Harvard GDP npz file")
    parser.add_argument("--out", type=str, help="Output directory for single file conversion")
    args = parser.parse_args()
    main(npz=args.npz, out=args.out)
