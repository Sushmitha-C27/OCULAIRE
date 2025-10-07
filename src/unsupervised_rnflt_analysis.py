import os, glob, json
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

def load_metrics(metrics_dir="/content/OCULAIRE/results/metrics"):
    records = []
    for fpath in glob.glob(os.path.join(metrics_dir, "*_rnflt_metrics.json")):
        with open(fpath, "r") as f:
            d = json.load(f)
            records.append({
                "case_id": d["case_id"],
                "mean": d["mean_rnflt"],
                "std": d["std_rnflt"],
                "min": d["min_rnflt"],
                "max": d["max_rnflt"]
            })
    df = pd.DataFrame(records)
    print(f"✅ Loaded {len(df)} RNFLT cases.")
    return df

def analyze_clusters(df, n_clusters=2):
    # Standardize data
    X = df[["mean", "std", "min", "max"]].values
    X_scaled = StandardScaler().fit_transform(X)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["cluster"] = clusters
    
    # Cluster statistics
    cluster_stats = df.groupby("cluster")[["mean", "std", "min", "max"]].mean().reset_index()
    print("\n📊 Cluster Summary (Average RNFLT per cluster):")
    print(cluster_stats)
    
    # Determine which cluster corresponds to thinner RNFL (likely glaucoma)
    thin_cluster = cluster_stats.loc[cluster_stats["mean"].idxmin(), "cluster"]
    thick_cluster = cluster_stats.loc[cluster_stats["mean"].idxmax(), "cluster"]
    print(f"\n🔎 Interpretation:")
    print(f" - Cluster {thin_cluster} → thinner RNFL (potential glaucoma group)")
    print(f" - Cluster {thick_cluster} → thicker RNFL (likely healthy group)")

    # Evaluate cluster quality
    sil_score = silhouette_score(X_scaled, clusters)
    db_score = davies_bouldin_score(X_scaled, clusters)
    print(f"\n🧠 Cluster Quality Metrics:")
    print(f" - Silhouette Score: {sil_score:.3f} (closer to 1 = better separation)")
    print(f" - Davies–Bouldin Index: {db_score:.3f} (closer to 0 = better separation)")

    # Visualization
    plt.figure(figsize=(7,6))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="viridis", s=50)
    plt.title("RNFLT Feature Clusters (PCA Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    # Save results
    os.makedirs("/content/OCULAIRE/results/analysis", exist_ok=True)
    df.to_csv("/content/OCULAIRE/results/analysis/rnflt_clusters.csv", index=False)
    cluster_stats.to_csv("/content/OCULAIRE/results/analysis/rnflt_cluster_summary.csv", index=False)

    print("\n✅ Saved:")
    print(" - Cluster assignments → results/analysis/rnflt_clusters.csv")
    print(" - Cluster summary → results/analysis/rnflt_cluster_summary.csv")

if __name__ == "__main__":
    df = load_metrics()
    analyze_clusters(df, n_clusters=2)

