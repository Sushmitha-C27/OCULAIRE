import numpy as np, os, json
from tqdm import tqdm

def extract_rnflt_metrics(input_dir="/content/OCULAIRE/data/raw/RNFLT",
                          output_dir="/content/OCULAIRE/results/metrics"):
    os.makedirs(output_dir, exist_ok=True)
    npz_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npz")])

    for fname in tqdm(npz_files, desc="Extracting RNFLT metrics"):
        fpath = os.path.join(input_dir, fname)
        case_id = os.path.splitext(fname)[0]
        data = np.load(fpath, allow_pickle=True)
        rnflt_map = data["volume"] if "volume" in data else data[data.files[0]]

        # flatten and compute stats
        values = rnflt_map.flatten().astype(float)
        mean_val = float(np.nanmean(values))
        std_val  = float(np.nanstd(values))
        min_val  = float(np.nanmin(values))
        max_val  = float(np.nanmax(values))

        metrics = {
            "case_id": case_id,
            "mean_rnflt": round(mean_val, 3),
            "std_rnflt": round(std_val, 3),
            "min_rnflt": round(min_val, 3),
            "max_rnflt": round(max_val, 3),
            "map_shape": rnflt_map.shape
        }

        with open(os.path.join(output_dir, f"{case_id}_rnflt_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"✅ Metrics extracted for {len(npz_files)} RNFLT cases → {output_dir}")

if __name__ == "__main__":
    extract_rnflt_metrics()
