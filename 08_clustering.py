from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "USSeatBelts.csv"
if not DATA_PATH.exists() and (SCRIPT_DIR / "USSeatBelts.csv").exists():
    DATA_PATH = SCRIPT_DIR / "USSeatBelts.csv"

def load_seatbelts():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find the dataset at {DATA_PATH.resolve()}. "
            "Place USSeatBelts.csv in this folder."
        )
    return pd.read_csv(DATA_PATH)


def show_data(df):
    print("\nSHAPE:", df.shape)
    print("\nCOLUMNS:", list(df.columns))

    print("\nDATA INFORMATION:")
    df.info()

    print("\nFIRST SIX ROWS:\n", df.head(6))
    print("\nLAST SIX ROWS:\n", df.tail(6))
    print("\nSUMMARY STATISTICS:\n", df.describe(include="all").T)


def display_assignments(results, title, output_path):
    print(f"\n{title}")
    print("-" * 45)
    print("Total observations:", len(results))
    print("Number of clusters:", results["Cluster"].nunique())

    print("\nCLUSTER COUNTS:")
    print(results["Cluster"].value_counts().sort_index().to_string())

    print("\nFIRST 20 CLUSTER ASSIGNMENTS:")
    print(results.head(20).to_string(index=False))

    print("\nLAST 10 CLUSTER ASSIGNMENTS:")
    print(results.tail(10).to_string(index=False))

    results.to_csv(output_path, index=False)
    print("\nComplete assignments saved to:")
    print(output_path)


df = load_seatbelts()
show_data(df)

# Check missing values
print("\nMISSING VALUES IN EACH COLUMN:")
print(df.isnull().sum().to_string())
print("\nTOTAL MISSING VALUES:", df.isnull().sum().sum())

# Display distributions of relevant categorical variables
for col in ["state", "speed65", "speed70", "drinkage", "alcohol", "enforce"]:
    if col in df.columns:
        print(f"\nDISTRIBUTION OF {col.upper()}:")
        print(df[col].value_counts(dropna=False).to_string())

# Select numerical features
num = df.select_dtypes(include=np.number).drop(
    columns=["rownames"], errors="ignore"
)

print("\nNUMERICAL FEATURES:")
print(list(num.columns))

# Remove rows with missing numerical values
rows_before = len(num)
num = num.dropna()
rows_dropped = rows_before - len(num)

print("\nROWS BEFORE REMOVING MISSING VALUES:", rows_before)
print("ROWS DROPPED:", rows_dropped)
print("ROWS USED FOR CLUSTERING:", len(num))

if len(num) < 3:
    raise ValueError("At least 3 complete observations are required.")

# Standardize numerical features
scaler = StandardScaler()
scaled = scaler.fit_transform(num)

scaled_df = pd.DataFrame(scaled, columns=num.columns, index=num.index)

print("\nSTANDARDIZED DATA (FIRST SIX ROWS):")
print(scaled_df.head(6).to_string())

# --------------------------------------------------
# K-MEANS CLUSTERING
# --------------------------------------------------

# Apply K-Means with K = 2
km2 = KMeans(n_clusters=2, n_init=25, random_state=123)
km2_labels = km2.fit_predict(scaled)

print("\nK-MEANS CLUSTERING (K = 2)")
print("-" * 45)

print("\nCLUSTER CENTERS (STANDARDIZED FEATURES):")
print(pd.DataFrame(
    km2.cluster_centers_, columns=num.columns
).to_string(index=False))

print("\nCLUSTER CENTERS (ORIGINAL UNITS):")
print(pd.DataFrame(
    scaler.inverse_transform(km2.cluster_centers_),
    columns=num.columns
).to_string(index=False))

print("\nCLUSTER COUNTS:")
print(pd.Series(km2_labels).value_counts().sort_index().to_string())

km2_results = pd.DataFrame({
    "Observation": num.index,
    "Cluster": km2_labels
})

display_assignments(
    km2_results,
    "K-MEANS ASSIGNMENTS (K = 2)",
    SCRIPT_DIR / "kmeans_k2_assignments.csv"
)

# Calculate WCSS for K = 1 to 10
max_k = min(10, len(scaled))
ks = range(1, max_k + 1)
wcss = []

for k in ks:
    model = KMeans(n_clusters=k, n_init=25, random_state=123)
    model.fit(scaled)
    wcss.append(model.inertia_)

# Print numerical WCSS scores
print("\nWCSS SCORES:")
print("-" * 45)
print(f"{'K':<10}{'WCSS':>15}")
print("-" * 25)

for k, score in zip(ks, wcss):
    print(f"{k:<10}{score:>15.4f}")

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(list(ks), wcss, marker="o")
plt.xticks(list(ks))
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal K")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Select K after examining the elbow plot
# Change this value if the elbow suggests another K
optimal_k = min(3, len(scaled))

print(f"\nSELECTED K FOR FINAL K-MEANS MODEL: {optimal_k}")

final_kmeans = KMeans(
    n_clusters=optimal_k,
    n_init=25,
    random_state=123
)
final_labels = final_kmeans.fit_predict(scaled)

print("\nFINAL K-MEANS CLUSTER CENTERS:")
print(pd.DataFrame(
    final_kmeans.cluster_centers_,
    columns=num.columns
).to_string(index=False))

print("\nFINAL K-MEANS CLUSTER COUNTS:")
print(pd.Series(final_labels).value_counts().sort_index().to_string())

final_results = pd.DataFrame({
    "Observation": num.index,
    "Cluster": final_labels
})

display_assignments(
    final_results,
    f"FINAL K-MEANS ASSIGNMENTS (K = {optimal_k})",
    SCRIPT_DIR / "kmeans_final_assignments.csv"
)

# --------------------------------------------------
# HIERARCHICAL CLUSTERING
# --------------------------------------------------

# Calculate Euclidean distances
euclidean_distances = pdist(scaled, metric="euclidean")

print("\nHIERARCHICAL CLUSTERING")
print("-" * 45)
print("Distance metric: Euclidean")
print("Linkage method: Ward")
print("Number of pairwise distances:", len(euclidean_distances))
print("First 10 Euclidean distances:", np.round(euclidean_distances[:10], 4))

# Perform Ward hierarchical clustering using all observations
Z = linkage(scaled, method="ward", metric="euclidean")

# Plot dendrogram
plt.figure(figsize=(16, 7))
dendrogram(Z, no_labels=True)
plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)")
plt.xlabel("Observations")
plt.ylabel("Ward Distance")
plt.tight_layout()
plt.show()

# Divide observations into 3 hierarchical clusters
hc_labels = fcluster(Z, t=3, criterion="maxclust")

hc_results = pd.DataFrame({
    "Observation": num.index,
    "Cluster": hc_labels
})

display_assignments(
    hc_results,
    "HIERARCHICAL CLUSTER ASSIGNMENTS (3 CLUSTERS)",
    SCRIPT_DIR / "hierarchical_cluster_assignments.csv"
)

# Cross-check using sklearn AgglomerativeClustering
hc_model = AgglomerativeClustering(n_clusters=3, linkage="ward")
hc_sklearn_labels = hc_model.fit_predict(scaled) + 1

print("\nAGGLOMERATIVE CLUSTERING COUNTS (SKLEARN):")
print(pd.Series(hc_sklearn_labels).value_counts().sort_index().to_string())
