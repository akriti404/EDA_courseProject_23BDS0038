from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "USSeatBelts.csv"
if not DATA_PATH.exists() and (SCRIPT_DIR / "USSeatBelts.csv").exists():
    DATA_PATH = SCRIPT_DIR / "USSeatBelts.csv"

def load_seatbelts():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH.resolve()}. "
            "Place USSeatBelts.csv or USSeatBelts(5).csv in this folder."
        )
    return pd.read_csv(DATA_PATH)

def show_data(df):
    print("\nDATASET DIMENSIONS:", df.shape)
    print("\nCOLUMNS:", list(df.columns))
    print("\nDATA INFORMATION:")
    df.info()
    print("\nFIRST SIX ROWS:")
    print(df.head(6).to_string(index=False))
    print("\nSUMMARY STATISTICS:")
    print(df.describe(include="all").T.to_string())

df = load_seatbelts()
show_data(df)

# Select numerical variables and exclude row identifiers
numeric = df.select_dtypes(include=np.number).copy()
numeric = numeric.drop(columns=["rownames"], errors="ignore")
numeric = numeric.dropna(axis=1, how="all")

print("\nMISSING VALUES BEFORE IMPUTATION:")
print(numeric.isnull().sum().to_string())

# Replace missing values with column medians
numeric = numeric.fillna(numeric.median())

# Remove constant columns
numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]

if numeric.shape[1] < 2:
    raise ValueError("PCA requires at least two non-constant numerical variables.")

# Standardize the data
scaler = StandardScaler()
data_scaled = pd.DataFrame(
    scaler.fit_transform(numeric),
    columns=numeric.columns,
    index=numeric.index
)

print("\nCLEANED SCALED DIMENSIONS:", data_scaled.shape)
print("\nSTANDARDIZED DATA (FIRST SIX ROWS):")
print(data_scaled.head(6).round(4).to_string())

print("\nSTANDARDIZED DATA SUMMARY:")
print(data_scaled.describe().round(4).to_string())

# Perform PCA
pca = PCA()
scores = pca.fit_transform(data_scaled)

variance = pca.explained_variance_ratio_ * 100
cumulative_variance = np.cumsum(variance)

# PCA summary table
pca_summary = pd.DataFrame({
    "Component": [f"PC{i + 1}" for i in range(len(variance))],
    "Eigenvalue": pca.explained_variance_,
    "Variance_Percent": variance,
    "Cumulative_Percent": cumulative_variance
})

print("\nPCA SUMMARY TABLE:")
print(pca_summary.round(4).to_string(index=False))

# Scree plot with y-axis from 0 to 60
plt.figure(figsize=(9, 5))
components = np.arange(1, len(variance) + 1)

plt.bar(
    components,
    variance,
    edgecolor="black"
)

for i, value in enumerate(variance):
    if value <= 60:
        plt.text(
            i + 1,
            value + 0.7,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8
        )

plt.xticks(components)
plt.yticks(np.arange(0, 61, 10))
plt.ylim(0, 60)
plt.xlabel("Principal Components")
plt.ylabel("Percentage of Variance Explained")
plt.title("Scree Plot of Principal Components")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()

# Determine number of PCs explaining at least 80% variance
num_pc = np.argmax(cumulative_variance >= 80) + 1

print(f"\nNUMBER OF PCs REQUIRED FOR AT LEAST 80% VARIANCE: {num_pc}")
print(f"CUMULATIVE VARIANCE EXPLAINED: {cumulative_variance[num_pc - 1]:.2f}%")

# PCA loadings for the first two components
loadings = pca.components_.T[:, :2]

# Biplot-style view: scores and variable loading arrows
fig, ax = plt.subplots(figsize=(11, 8))

ax.scatter(
    scores[:, 0],
    scores[:, 1],
    color="blue",
    alpha=0.45,
    s=20
)

# Scale arrows relative to score range
score_range = max(
    np.ptp(scores[:, 0]),
    np.ptp(scores[:, 1]),
    1
)
arrow_scale = score_range * 0.18

for i, name in enumerate(data_scaled.columns):
    dx = loadings[i, 0] * arrow_scale
    dy = loadings[i, 1] * arrow_scale

    ax.annotate(
        "",
        xy=(dx, dy),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2)
    )

    ax.annotate(
        name,
        (dx, dy),
        xytext=(4, 4),
        textcoords="offset points",
        color="darkred",
        fontsize=8
    )

ax.axhline(0, color="grey", linewidth=0.7)
ax.axvline(0, color="grey", linewidth=0.7)
ax.set_title("PCA Biplot of US Seat Belt Dataset")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# Save PCA scores for all principal components
pca_scores = pd.DataFrame(
    scores,
    columns=[f"PC{i + 1}" for i in range(scores.shape[1])],
    index=df.index
)

print("\nFIRST SIX PCA SCORES:")
print(pca_scores.head(6).round(4).to_string())

# PC1 versus PC2 scatter plot
plt.figure(figsize=(9, 6))
plt.scatter(
    pca_scores["PC1"],
    pca_scores["PC2"],
    color="blue",
    marker="o",
    alpha=0.7
)

plt.title("PCA: PC1 vs PC2 - 23BDS0038")
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()