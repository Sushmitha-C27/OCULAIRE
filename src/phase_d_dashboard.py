
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from IPython.display import display, Markdown
from google.colab import files

# ========== Load cluster model ==========
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

# ========== Upload RNFLT ==========
def upload_and_process_npz():
    print("📤 Please upload an RNFLT .npz file to analyze:")
    uploaded = files.upload()
    if not uploaded:
        print("⚠️ No file uploaded."); return None, None, None
    fname = list(uploaded.keys())[0]
    npz = np.load(fname, allow_pickle=True)
    rnflt_map = npz["volume"] if "volume" in npz else npz[npz.files[0]]
    values = rnflt_map.flatten().astype(float)
    metrics = {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values))
    }
    print(f"✅ Metrics extracted: {metrics}")
    return fname, rnflt_map, metrics

# ========== Quadrant Metrics ==========
def compute_quadrant_metrics(rnflt_map):
    h, w = rnflt_map.shape
    half_h, half_w = h // 2, w // 2
    return {
        "Superior": np.nanmean(rnflt_map[:half_h,:]),
        "Inferior": np.nanmean(rnflt_map[half_h:,:]),
        "Nasal":    np.nanmean(rnflt_map[:,:half_w]),
        "Temporal": np.nanmean(rnflt_map[:,half_w:])
    }

# ========== Cluster Avg Quadrant ==========
def load_cluster_quadrant_summary():
    path = "/content/OCULAIRE/results/analysis/rnflt_cluster_quadrant_summary.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        print("⚠️ Cluster quadrant summary not found. Run rnflt_cluster_quadrant_summary.py.")
        return None

# ========== Classification & Visualization ==========
def classify_and_visualize(fname, rnflt_map, metrics, scaler, kmeans,
                            thin_cluster, thick_cluster, cluster_quad_summary):
    X_new = np.array([[metrics["mean"],metrics["std"],metrics["min"],metrics["max"]]])
    cluster = int(kmeans.predict(scaler.transform(X_new))[0])
    label = "Healthy-like" if cluster == thick_cluster else "Glaucoma-like"

    display(Markdown(f"## 🩺 Classification Result"))
    display(Markdown(f"**File:** {fname}"))
    display(Markdown(f"**Predicted Cluster:** {cluster}"))
    display(Markdown(f"**Interpretation:** {label}"))
    display(Markdown(f"**Mean RNFLT:** {metrics['mean']:.2f} µm"))

    # --- Quadrant metrics ---
    quads = compute_quadrant_metrics(rnflt_map)
    display(Markdown("### 🧭 Quadrant RNFLT Thickness (µm)"))
    df_case = pd.DataFrame(quads, index=["Uploaded"])
    df_cluster = cluster_quad_summary.set_index("cluster")
    healthy_avg = df_cluster.loc[thick_cluster, ["Superior","Inferior","Nasal","Temporal"]]
    glaucoma_avg = df_cluster.loc[thin_cluster, ["Superior","Inferior","Nasal","Temporal"]]

    print(df_case.T)
    # --- Visual comparison ---
    plt.figure(figsize=(8,4))
    for i,q in enumerate(["Superior","Inferior","Nasal","Temporal"]):
        case_val = quads[q]; healthy = healthy_avg[q]; glaucoma = glaucoma_avg[q]
        color = "#D62728" if case_val < healthy-8 else "#2CA02C"
        plt.bar(i, case_val, color=color)
        plt.plot([i-0.3,i+0.3],[healthy,healthy],'k--',label="Healthy ref" if i==0 else "")
        plt.plot([i-0.3,i+0.3],[glaucoma,glaucoma],'b:',label="Glaucoma ref" if i==0 else "")
    plt.xticks(range(4),["Superior","Inferior","Nasal","Temporal"])
    plt.ylabel("Thickness (µm)")
    plt.title(f"Quadrant RNFLT Comparison – {label}")
    plt.legend()
    plt.show()

    # --- RNFLT Map ---
    plt.imshow(rnflt_map,cmap='turbo'); plt.colorbar(label="RNFLT (µm)")
    plt.title(f"RNFLT Map – {label}"); plt.axis('off'); plt.show()

# ========== Runner ==========
def run_dashboard():
    df, scaler, kmeans, thin_cluster, thick_cluster, rnflt_raw = load_cluster_model()
    cluster_quad_summary = load_cluster_quadrant_summary()
    fname, rnflt_map, metrics = upload_and_process_npz()
    if metrics:
        classify_and_visualize(fname, rnflt_map, metrics, scaler, kmeans,
                               thin_cluster, thick_cluster, cluster_quad_summary)
