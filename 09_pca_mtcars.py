import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load mtcars directly from the online R dataset
df = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data

print("DATASET SHAPE:", df.shape)
print("\nFIRST SIX ROWS:")
print(df.head(6).to_string())

# Select the six variables specified in the assignment
cols = ["mpg", "disp", "hp", "drat", "wt", "qsec"]
data = df[cols].copy()

print("\nSELECTED DATA DIMENSIONS:", data.shape)
print("\nSUMMARY STATISTICS:")
print(data.describe().round(4).to_string())

# Standardize the data
scaler = StandardScaler()
scaled = scaler.fit_transform(data)

scaled_df = pd.DataFrame(
    scaled, columns=cols, index=data.index
)

print("\nSTANDARDIZED DATA (FIRST SIX ROWS):")
print(scaled_df.head(6).round(4).to_string())

# Perform PCA
pca = PCA()
scores = pca.fit_transform(scaled)

eigenvalues = pca.explained_variance_
variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance)

# Display PCA variance table
table = pd.DataFrame({
    "Principal_Component": [
        f"PC{i + 1}" for i in range(len(variance))
    ],
    "Eigenvalue": eigenvalues,
    "Proportion_Variance": variance,
    "Percentage_Variance": variance * 100,
    "Cumulative_Percentage": cumulative_variance * 100
})

print("\nPCA VARIANCE TABLE:")
print(table.round(4).to_string(index=False))

# Number of principal components needed for at least 80% variance
num_pc = np.argmax(cumulative_variance >= 0.80) + 1

print(
    f"\nPrincipal components needed for at least 80% variance: {num_pc}"
)
print(
    f"Cumulative variance explained: "
    f"{cumulative_variance[num_pc - 1] * 100:.2f}%"
)

# Scree plot
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(eigenvalues) + 1),
    eigenvalues,
    marker="o"
)
plt.xticks(range(1, len(eigenvalues) + 1))
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalue")
plt.title("Scree Plot - mtcars")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# PCA loadings
loadings = pd.DataFrame(
    pca.components_.T,
    index=cols,
    columns=[f"PC{i + 1}" for i in range(len(cols))]
)

print("\nPCA LOADINGS:")
print(loadings.round(4).to_string())

print("\nPC1 LOADINGS (SORTED BY ABSOLUTE MAGNITUDE):")
print(loadings["PC1"].sort_values(
    key=abs, ascending=False
).round(4).to_string())

print("\nPC2 LOADINGS (SORTED BY ABSOLUTE MAGNITUDE):")
print(loadings["PC2"].sort_values(
    key=abs, ascending=False
).round(4).to_string())

# PCA scores plot with observation labels
plt.figure(figsize=(9, 6))
plt.scatter(scores[:, 0], scores[:, 1], s=45)

for i, label in enumerate(data.index):
    plt.annotate(
        str(label),
        (scores[i, 0], scores[i, 1]),
        fontsize=8,
        xytext=(4, 4),
        textcoords="offset points"
    )

plt.axhline(0, color="grey", linewidth=0.7)
plt.axvline(0, color="grey", linewidth=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Scores Plot - mtcars")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()

# PCA biplot: observation scores and variable-loading arrows
plt.figure(figsize=(10, 7))
plt.scatter(scores[:, 0], scores[:, 1], s=35, alpha=0.7)

for i, label in enumerate(data.index):
    plt.annotate(
        str(label),
        (scores[i, 0], scores[i, 1]),
        fontsize=7,
        xytext=(3, 3),
        textcoords="offset points"
    )

# Scale arrows for readability; arrows show variable directions
arrow_scale = min(
    np.max(np.abs(scores[:, :2])) / 2,
    3
)

for i, variable in enumerate(cols):
    x_loading = pca.components_[0, i] * arrow_scale
    y_loading = pca.components_[1, i] * arrow_scale

    plt.arrow(
        0, 0,
        x_loading, y_loading,
        color="red",
        head_width=0.06,
        length_includes_head=True
    )

    plt.text(
        x_loading * 1.08,
        y_loading * 1.08,
        variable,
        color="red",
        fontsize=10,
        ha="center",
        va="center"
    )

plt.axhline(0, color="grey", linewidth=0.7)
plt.axvline(0, color="grey", linewidth=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Biplot - mtcars")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()