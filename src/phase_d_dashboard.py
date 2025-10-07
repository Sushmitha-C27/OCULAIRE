import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from IPython.display import display, Markdown
from google.colab import files

# ==========================================================
# 1. Load cluster model
# ==========================================================
def load_cluster_model():
    cluster_csv = "/content/OCULAIRE/results/analysis/rnflt_clusters.csv"
    rnflt_raw = "/content/OCULAIRE/data/raw/RNFLT"
    df = pd.read_csv(cluster_csv)
    scaler = StandardScaler().fit(df[["mean","std","min","max"]])
    kmeans = KMeans(n_clusters=df["cluster"].nunique(), random_state=42, n_init=10)
    kmeans.fit(scaler.transform(df[["mean","std","min","max"]]))
    cluster_summary = df.groupby("cluster")[["mean"]].mean().sort_values("mean")
    thin_cluster = int(cluster_summary.index[0])
    thick_cluster = int(cluster_summary.index[-1])
    print(f"✅ Loaded {len(df)} RNFLT cases.")
    return df, scaler, kmeans, thin_cluster, thick_cluster, rnflt_raw

# ==========================================================
# 2. Upload RNFLT map
# ==========================================================
def upload_and_process_npz():
    print("📤 Please upload an RNFLT .npz file to analyze:")
    uploaded = files.upload()
    if not uploaded:
        print("⚠️ No file uploaded."); return None, None, None
    fname = list(uploaded.keys())[0]
    npz = np.load(fname, allow_pickle=True)
    rnflt_map = npz["volume"] if "volume" in npz else npz[npz.files[0]]
    vals = rnflt_map.flatten().astype(float)
    metrics = {
        "mean": float(np.nanmean(vals)),
        "std": float(np.nanstd(vals)),
        "min": float(np.nanmin(vals)),
        "max": float(np.nanmax(vals))
    }
    print(f"✅ Metrics extracted: {metrics}")
    return fname, rnflt_map, metrics

# ==========================================================
# 3. Compute average maps for clusters
# ==========================================================
def compute_average_maps(df, rnflt_raw):
    avg_maps = {}
    for cid in df["cluster"].unique():
        subset = df[df["cluster"] == cid]["case_id"]
        stack = []
        for case in subset:
            path = os.path.join(rnflt_raw, f"{case}.npz")
            if os.path.exists(path):
                data = np.load(path, allow_pickle=True)
                arr = data["volume"] if "volume" in data else data[data.files[0]]
                stack.append(arr)
        if stack:
            avg_maps[cid] = np.nanmean(np.stack(stack), axis=0)
    print(f"✅ Computed average maps for {len(avg_maps)} clusters.")
    return avg_maps

# ==========================================================
# 4. Compute risk & severity + save image
# ==========================================================
def compute_risk_map(rnflt_map, healthy_avg, threshold=-10, save_path=None, fname=None):
    if healthy_avg is None:
        print("⚠️ No healthy average map found. Using global mean instead.")
        healthy_avg = np.full_like(rnflt_map, np.nanmean(rnflt_map))
    diff = rnflt_map - healthy_avg
    risk = np.where(diff < threshold, diff, np.nan)
    # ---- Severity (% of pixels below threshold) ----
    total_pixels = np.isfinite(diff).sum()
    risky_pixels = np.isfinite(risk).sum()
    severity = (risky_pixels / total_pixels) * 100 if total_pixels > 0 else np.nan
    # ---- Auto-save risk map ----
    if save_path and fname:
        os.makedirs(save_path, exist_ok=True)
        plt.imsave(os.path.join(save_path, f"{os.path.splitext(fname)[0]}_risk.png"),
                   risk, cmap="hot")
    return diff, risk, severity

# ==========================================================
# 5. Classification + visualization
# ==========================================================
def classify_and_visualize(fname, rnflt_map, metrics, scaler, kmeans,
                            thin_cluster, thick_cluster, avg_maps):
    X_new = np.array([[metrics["mean"],metrics["std"],metrics["min"],metrics["max"]]])
    cluster = int(kmeans.predict(scaler.transform(X_new))[0])
    label = "Healthy-like" if cluster == thick_cluster else "Glaucoma-like"

    display(Markdown(f"## 🩺 Classification Result"))
    display(Markdown(f"**File:** {fname}"))
    display(Markdown(f"**Predicted Cluster:** {cluster}"))
    display(Markdown(f"**Interpretation:** {label}"))
    display(Markdown(f"**Mean RNFLT:** {metrics['mean']:.2f} µm"))

    # --- RNFLT Map ---
    plt.figure(figsize=(6,5))
    plt.imshow(rnflt_map, cmap='turbo')
    plt.title(f"RNFLT Map – {label}")
    plt.colorbar(label="Thickness (µm)")
    plt.axis('off')
    plt.show()

    # --- Compute Risk Map & Severity ---
    healthy_avg = avg_maps.get(thick_cluster) if avg_maps else None
    diff, risk, severity = compute_risk_map(
        rnflt_map, healthy_avg, threshold=-10,
        save_path="/content/OCULAIRE/results/risk_maps", fname=fname)

    display(Markdown(f"### ⚠️ Severity Score: **{severity:.2f}%** of area shows thinning < –10 µm"))

    # --- Trio of Maps ---
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(healthy_avg, cmap='turbo')
    plt.title("Average Healthy RNFLT" if healthy_avg is not None else "Fallback RNFLT")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(diff, cmap='bwr', vmin=-20, vmax=20)
    plt.title("Difference Map (Case – Healthy)")
    plt.colorbar(label="Δ Thickness (µm)")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(risk, cmap='hot')
    plt.title("Risk Map (Thinner Zones)")
    plt.colorbar(label="Δ Thickness (µm)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# ==========================================================
# 6. Runner
# ==========================================================
def run_dashboard():
    df, scaler, kmeans, thin_cluster, thick_cluster, rnflt_raw = load_cluster_model()
    avg_maps = compute_average_maps(df, rnflt_raw)
    fname, rnflt_map, metrics = upload_and_process_npz()
    if metrics:
        classify_and_visualize(fname, rnflt_map, metrics, scaler,
                               kmeans, thin_cluster, thick_cluster, avg_maps)
