
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, pearsonr, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "USSeatBelts.csv"

if not DATA_PATH.exists() and (SCRIPT_DIR / "USSeatBelts.csv").exists():
    DATA_PATH = SCRIPT_DIR / "USSeatBelts.csv"

def load_seatbelts():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find the dataset at {DATA_PATH}.\n"
            "Place USSeatBelts.csv"
            "in the same folder as this script."
        )
    return pd.read_csv(DATA_PATH)


def show_data(df):
    print("\nDATASET INFORMATION")
    print("\nShape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nData types:")
    df.info()

    print("\nFirst six rows:")
    print(df.head(6).to_string())

    print("\nLast six rows:")
    print(df.tail(6).to_string())

    print("\nSummary statistics:")
    print(df.describe(include="all").T.to_string())


df = load_seatbelts()
show_data(df)

required_columns = [
    "fatalities", "seatbelt", "speed65", "speed70",
    "miles", "income", "age"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")


# 1-3. DESCRIPTIVE STATISTICS

x = pd.to_numeric(df["fatalities"], errors="coerce").dropna()

stats = {
    "Mean": x.mean(),
    "Median": x.median(),
    "Minimum": x.min(),
    "Maximum": x.max(),
    "Range": x.max() - x.min(),
    "Variance": x.var(),
    "Standard Deviation": x.std(),
    "IQR": x.quantile(0.75) - x.quantile(0.25),
    "Skewness": skew(x),
    "Kurtosis": kurtosis(x)
}

print("\nFATALITIES SUMMARY STATISTICS")
print(pd.Series(stats).to_string())


# 4. FREQUENCY AND RELATIVE FREQUENCY DISTRIBUTIONS

freq = df["seatbelt"].value_counts(dropna=False)
relative_freq = freq / freq.sum()

print("\nSEATBELT FREQUENCY DISTRIBUTION")
print(freq.to_string())

print("\nSEATBELT RELATIVE FREQUENCY DISTRIBUTION")
print(relative_freq.to_string())

print("\nSEATBELT PERCENTAGE DISTRIBUTION")
print((relative_freq * 100).round(2).to_string())

freq.plot(
    kind="pie",
    autopct="%1.1f%%",
    figsize=(7, 7),
    title="Seatbelt Distribution"
)
plt.ylabel("")
plt.tight_layout()
plt.show()

freq.plot(
    kind="bar",
    figsize=(8, 5),
    title="Seatbelt Frequency Distribution"
)
plt.xlabel("Seatbelt")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# 5. NUMERICAL FREQUENCY TABLE, HISTOGRAM AND BOX PLOT

bins = pd.cut(x, bins=10)

print("\nFATALITIES FREQUENCY DISTRIBUTION BY INTERVAL")
print(bins.value_counts().sort_index().to_string())

plt.figure(figsize=(8, 5))
plt.hist(x, bins=10, edgecolor="black")
plt.title("Histogram of Fatalities")
plt.xlabel("Fatalities")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.boxplot(y=x)
plt.title("Box Plot of Fatalities")
plt.ylabel("Fatalities")
plt.tight_layout()
plt.show()


# 6-7. CONTINGENCY TABLE, PROPORTIONS AND STACKED BAR PLOT

ct = pd.crosstab(df["speed65"], df["speed70"])

print("\nCONTINGENCY TABLE: SPEED65 VS SPEED70")
print(ct.to_string())

print("\nROW PROPORTIONS")
print(pd.crosstab(
    df["speed65"], df["speed70"], normalize="index"
).round(4).to_string())

print("\nCOLUMN PROPORTIONS")
print(pd.crosstab(
    df["speed65"], df["speed70"], normalize="columns"
).round(4).to_string())

ct.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5),
    title="Stacked Bar Plot: Speed65 vs Speed70"
)
plt.xlabel("Speed65")
plt.ylabel("Frequency")
plt.legend(title="Speed70")
plt.tight_layout()
plt.show()


# 8. GROUPED MEAN, MEDIAN AND STANDARD DEVIATION

df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce")

grouped_stats = df.groupby(
    "speed65", observed=False
)["fatalities"].agg(["mean", "median", "std"])

print("\nFATALITIES STATISTICS GROUPED BY SPEED65")
print(grouped_stats.to_string())


# 9. SIDE-BY-SIDE BOX PLOT AND VIOLIN PLOT

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="speed65", y="fatalities")
plt.title("Fatalities by Speed65")
plt.xlabel("Speed65")
plt.ylabel("Fatalities")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x="speed65", y="fatalities")
plt.title("Violin Plot of Fatalities by Speed65")
plt.xlabel("Speed65")
plt.ylabel("Fatalities")
plt.tight_layout()
plt.show()


# 10. PEARSON AND SPEARMAN CORRELATIONS

paired = df[["miles", "fatalities"]].apply(
    pd.to_numeric, errors="coerce"
).dropna()

if len(paired) >= 2:
    print("\nPEARSON CORRELATION")
    print(pearsonr(paired["miles"], paired["fatalities"]))

    print("\nSPEARMAN CORRELATION")
    print(spearmanr(paired["miles"], paired["fatalities"]))

plt.figure(figsize=(8, 5))
sns.regplot(data=paired, x="miles", y="fatalities")
plt.title("Miles vs Fatalities with Linear Regression")
plt.xlabel("Miles")
plt.ylabel("Fatalities")
plt.tight_layout()
plt.show()


# 11. TWO CATEGORICAL VARIABLES AND ONE NUMERICAL VARIABLE

grouped = df.groupby(
    ["speed65", "speed70"], observed=False
)["fatalities"].agg(["mean", "std", "count"]).reset_index()

print("\nMEAN AND STANDARD DEVIATION BY SPEED65 AND SPEED70")
print(grouped.to_string(index=False))

plt.figure(figsize=(9, 5))
sns.boxplot(
    data=df,
    x="speed65",
    y="fatalities",
    hue="speed70"
)
plt.title("Fatalities by Speed65 and Speed70")
plt.xlabel("Speed65")
plt.ylabel("Fatalities")
plt.tight_layout()
plt.show()

# Grouped mean bar chart with standard-deviation error bars
categories65 = list(df["speed65"].dropna().unique())
categories70 = list(df["speed70"].dropna().unique())

positions = np.arange(len(categories65))
width = 0.8 / max(len(categories70), 1)

fig, ax = plt.subplots(figsize=(9, 5))

for j, category70 in enumerate(categories70):
    subset = (
        grouped[grouped["speed70"] == category70]
        .set_index("speed65")
        .reindex(categories65)
    )

    offsets = positions - 0.4 + width / 2 + j * width

    ax.bar(
        offsets,
        subset["mean"],
        width=width,
        yerr=subset["std"],
        capsize=4,
        label=f"Speed70 = {category70}"
    )

ax.set_xticks(positions)
ax.set_xticklabels([str(category) for category in categories65])
ax.set_xlabel("Speed65")
ax.set_ylabel("Mean Fatalities")
ax.set_title("Mean Fatalities by Speed Categories with SD Error Bars")
ax.legend()
plt.tight_layout()
plt.show()


# 12. CATEGORY-COLORED SCATTER PLOT AND CATEGORY-WISE CORRELATIONS

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="miles",
    y="fatalities",
    hue="speed65",
    style="speed65"
)
plt.title("Miles vs Fatalities by Speed65 Category")
plt.xlabel("Miles")
plt.ylabel("Fatalities")
plt.tight_layout()
plt.show()

print("\nCORRELATIONS BY SPEED65 CATEGORY")

for category, group in df.groupby("speed65", observed=False):
    subset = group[["miles", "fatalities"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    print(f"\nSpeed65 = {category}")

    if len(subset) >= 2 and subset["miles"].nunique() > 1 and subset["fatalities"].nunique() > 1:
        print(
            "Pearson correlation:",
            subset["miles"].corr(
                subset["fatalities"], method="pearson"
            )
        )
        print(
            "Spearman correlation:",
            subset["miles"].corr(
                subset["fatalities"], method="spearman"
            )
        )
    else:
        print("Insufficient variation or observations for correlation.")


# 13. CORRELATION MATRIX AND HEATMAP

num = df[["miles", "fatalities", "income", "age"]].apply(
    pd.to_numeric, errors="coerce"
)

corr_matrix = num.corr()

print("\nCORRELATION MATRIX")
print(corr_matrix.round(4).to_string())

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# 14. PAIRS PLOT / SCATTERPLOT MATRIX

pairs_data = num.dropna()

pd.plotting.scatter_matrix(
    pairs_data,
    figsize=(10, 10),
    diagonal="hist",
    alpha=0.7
)

plt.suptitle("Pairs Plot of Numerical Variables", y=1.02)
plt.tight_layout()
plt.show()