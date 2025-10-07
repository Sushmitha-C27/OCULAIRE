
import os, json, numpy as np, pandas as pd
from tqdm import tqdm

def compute_quadrant_metrics(rnflt_map):
    """Return mean RNFLT for Superior, Inferior, Nasal, Temporal quadrants."""
    h, w = rnflt_map.shape
    half_h, half_w = h // 2, w // 2
    quadrants = {
        "Superior": np.nanmean(rnflt_map[:half_h, :]),
        "Inferior": np.nanmean(rnflt_map[half_h:, :]),
        "Nasal":    np.nanmean(rnflt_map[:, :half_w]),
        "Temporal": np.nanmean(rnflt_map[:, half_w:])
    }
    return quadrants

def rnflt_quadrant_analysis(
    input_dir="/content/OCULAIRE/data/raw/RNFLT",
    output_csv="/content/OCULAIRE/results/metrics/rnflt_quadrants.csv"
):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results = []

    for file in tqdm(sorted(os.listdir(input_dir)), desc="Processing RNFLT maps"):
        if not file.endswith(".npz"): continue
        case_id = os.path.splitext(file)[0]
        npz = np.load(os.path.join(input_dir, file), allow_pickle=True)
        rnflt_map = npz["volume"] if "volume" in npz else npz[npz.files[0]]
        q = compute_quadrant_metrics(rnflt_map)

        results.append({
            "case_id": case_id,
            "mean": float(np.nanmean(rnflt_map)),
            "std": float(np.nanstd(rnflt_map)),
            "min": float(np.nanmin(rnflt_map)),
            "max": float(np.nanmax(rnflt_map)),
            **q
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved quadrant metrics for {len(df)} cases → {output_csv}")
    return df

if __name__ == "__main__":
    rnflt_quadrant_analysis()
