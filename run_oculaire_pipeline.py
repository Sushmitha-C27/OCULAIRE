#!/usr/bin/env python3
"""
Top-level OCULAIRE runner script
- Converts .npz → PNG (skips if PNGs already exist)
- Trains model
- Evaluates & visualizes results
"""

import os
import argparse
from src.preprocessing.harvard_npz_adapter import convert_npz_to_png
from src.training.trainer import train_model, evaluate_model
from src.utils.visualization import plot_results


def process_folder(npz_folder, out_folder, key_name):
    """
    Convert all .npz files in a folder to PNGs if not already converted.
    """
    os.makedirs(out_folder, exist_ok=True)

    # Check if folder already contains PNGs
    existing_pngs = [f for f in os.listdir(out_folder) if f.endswith(".png")]
    if existing_pngs:
        print(f"✅ PNGs already exist in {out_folder}, skipping conversion.")
        return

    print(f"🔹 Converting {npz_folder} → {out_folder}")
    convert_npz_to_png(npz_folder, out_folder, key_name)


def main(npz=None, out=None, skip_conversion=False):
    # ---------------------------
    # Single-file mode
    # ---------------------------
    if npz and out:
        if not skip_conversion:
            print(f"🔹 Converting single file: {npz} → {out}")
            convert_npz_to_png(npz_folder=os.path.dirname(npz),
                               out_folder=out,
                               key_name="bscans")  # or "rnflt"
    else:
        # ---------------------------
        # Full dataset mode
        # ---------------------------
        RAW_BSCAN = "data/raw/Bscan"
        RAW_RNFLT = "data/raw/RNFLT"
        PROCESSED_BSCAN = "data/processed/Bscan"
        PROCESSED_RNFLT = "data/processed/RNFLT"

        if not skip_conversion:
            process_folder(RAW_BSCAN, PROCESSED_BSCAN, key_name="bscans")
            process_folder(RAW_RNFLT, PROCESSED_RNFLT, key_name="rnflt")
        else:
            print("⚡ Skipping conversion step, using existing PNGs.")

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
    parser.add_argument("--skip-conversion", action="store_true",
                        help="Skip converting .npz → PNG if PNGs already exist")
    args = parser.parse_args()

    main(npz=args.npz, out=args.out, skip_conversion=args.skip_conversion)

