from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "USSeatBelts.csv"
if not DATA_PATH.exists() and (SCRIPT_DIR / "USSeatBelts.csv").exists():
    DATA_PATH = SCRIPT_DIR / "USSeatBelts.csv"

def load_seatbelts():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH.resolve()}. Put USSeatBelts.csv in this folder "
            "or change DATA_PATH at the top of this script."
        )
    return pd.read_csv(DATA_PATH)

def show_data(df):
    print("\nSHAPE:", df.shape)
    print("\nCOLUMNS:", list(df.columns))
    print("\nINFO:")
    df.info()
    print("\nFIRST SIX ROWS:\n", df.head(6))
    print("\nLAST SIX ROWS:\n", df.tail(6))
    print("\nSUMMARY:\n", df.describe(include="all").T)

df = load_seatbelts()
show_data(df)
print("\nMISSING COUNTS\n", df.isna().sum())
print("\nMISSING PERCENTAGE\n", (df.isna().mean()*100).round(2))
print("\nROWS WITH MISSING VALUES\n", df[df.isna().any(axis=1)].head())

# Remove columns with more than 70% missing values.
clean = df.loc[:, df.isna().mean() <= .70].copy()
# Numeric imputations shown as separate strategies.
numeric_cols = clean.select_dtypes(include=np.number).columns
mean_data, median_data = clean.copy(), clean.copy()
for c in numeric_cols:
    if mean_data[c].isna().any() and mean_data[c].notna().any():
        mean_data[c] = mean_data[c].fillna(mean_data[c].mean())
    if median_data[c].isna().any() and median_data[c].notna().any():
        median_data[c] = median_data[c].fillna(median_data[c].median())
mode_data = clean.copy()
for c in mode_data.select_dtypes(exclude=np.number).columns:
    if mode_data[c].isna().any() and not mode_data[c].mode().empty:
        mode_data[c] = mode_data[c].fillna(mode_data[c].mode()[0])
print("\nMEAN IMPUTATION MISSING COUNTS\n", mean_data.isna().sum())
print("\nMEDIAN IMPUTATION MISSING COUNTS\n", median_data.isna().sum())
print("\nMODE IMPUTATION MISSING COUNTS\n", mode_data.isna().sum())
print("\nDUPLICATE ROW COUNT:", mode_data.duplicated().sum())
mode_data = mode_data.drop_duplicates()
print("ROWS AFTER DUPLICATE REMOVAL:", len(mode_data))

# Summary statistics for fatalities.
x = pd.to_numeric(mode_data.get("fatalities"), errors="coerce").dropna()
if len(x):
    print("\nFATALITIES SUMMARY\n", x.describe())
    print("Range:", x.max()-x.min(), "Variance:", x.var(), "SD:", x.std(),
          "IQR:", x.quantile(.75)-x.quantile(.25),
          "Skewness:", x.skew(), "Kurtosis:", x.kurtosis())

# Categorical frequency and relative frequency.
for col in ["seatbelt", "alcohol", "state"]:
    if col in mode_data:
        freq = mode_data[col].value_counts(dropna=False)
        print(f"\n{col.upper()} FREQUENCY\n", freq)
        print(f"{col.upper()} RELATIVE FREQUENCY\n", mode_data[col].value_counts(normalize=True, dropna=False))
        if col == "seatbelt":
            freq.plot(kind="pie", autopct="%1.1f%%", title="Seatbelt Distribution")
            plt.ylabel(""); plt.tight_layout(); plt.show()
            freq.plot(kind="bar", title="Seatbelt Distribution")
            plt.xlabel(col); plt.ylabel("Frequency"); plt.tight_layout(); plt.show()

if len(x):
    print("\nFATALITIES FREQUENCY TABLE\n", pd.cut(x, bins=10).value_counts().sort_index())
    x.plot(kind="hist", bins=10, title="Histogram of Fatalities", xlabel="Fatalities")
    plt.tight_layout(); plt.show()
    sns.boxplot(y=x); plt.title("Box Plot of Fatalities"); plt.tight_layout(); plt.show()
    # IQR outlier detection and filtering
    q1, q3 = x.quantile(.25), x.quantile(.75)
    iqr = q3-q1
    low, high = q1-1.5*iqr, q3+1.5*iqr
    outliers = mode_data[(mode_data["fatalities"] < low) | (mode_data["fatalities"] > high)]
    print("IQR LIMITS:", low, high, "\nOUTLIER COUNT:", len(outliers))
    no_outliers = mode_data[mode_data["fatalities"].between(low, high)].copy()
    sns.histplot(no_outliers["fatalities"], bins=30, kde=False, color="steelblue"); plt.title("Fatalities after IQR filtering"); plt.show()
    sns.histplot(no_outliers["fatalities"], bins=30, stat="density", kde=False, color="steelblue"); plt.title("Histogram with Density Curve"); plt.show()
    if "alcohol" in no_outliers:
        sns.violinplot(data=no_outliers, x="alcohol", y="fatalities"); plt.title("Fatalities by Alcohol"); plt.show()
    sns.kdeplot(data=no_outliers, x="income" if "income" in no_outliers else "fatalities", color="steelblue", fill=False); plt.title("Density Plot"); plt.show()

# Proportion/percentage and charts for categorical variables.
if "alcohol" in mode_data:
    counts = mode_data["alcohol"].value_counts()
    print("\nALCOHOL PROPORTIONS\n", counts/counts.sum())
    print("\nALCOHOL PERCENTAGES\n", counts/counts.sum()*100)
    counts.plot(kind="bar", title="Frequency of Alcohol Categories"); plt.ylabel("Frequency"); plt.show()
    (counts/counts.sum()*100).plot(kind="bar", title="Percentage Distribution of Alcohol"); plt.ylabel("Percentage"); plt.show()
    counts.plot(kind="pie", autopct="%1.1f%%", title="Pie Chart of Alcohol Categories"); plt.ylabel(""); plt.show()
if "state" in mode_data:
    mode_data["state"].value_counts().sort_values().plot(kind="barh", title="Frequency by State")
    plt.tight_layout(); plt.show()


# Additional transformations and plots represented in the original lab.
# Uppercase text values (demonstrated on a separate copy).
upper_data = mode_data.copy()
for c in upper_data.select_dtypes(include="object").columns:
    upper_data[c] = upper_data[c].str.upper()
print("\nUPPERCASE TEXT SAMPLE\n", upper_data.head())

# Boxplots and IQR outlier analysis for several numerical variables.
for col in ["fatalities", "income", "age"]:
    if col in mode_data:
        sns.boxplot(y=pd.to_numeric(mode_data[col], errors="coerce"))
        plt.title(f"Boxplot of {col.title()}")
        plt.tight_layout()
        plt.show()

# Small-sample dot plot equivalent.
if "fatalities" in mode_data:
    small = pd.to_numeric(mode_data["fatalities"], errors="coerce").dropna().head(30)
    plt.figure(figsize=(8,3))
    plt.scatter(small, np.zeros(len(small)), alpha=.75)
    plt.yticks([])
    plt.xlabel("Fatalities")
    plt.title("Dot Plot of Fatalities (first 30 valid observations)")
    plt.tight_layout()
    plt.show()

# Lollipop and dot charts of state frequencies.
if "state" in mode_data:
    state_freq = mode_data["state"].value_counts().sort_values()
    plt.figure(figsize=(10,5))
    plt.vlines(range(len(state_freq)), 0, state_freq.values)
    plt.scatter(range(len(state_freq)), state_freq.values)
    plt.xticks(range(len(state_freq)), state_freq.index, rotation=90)
    plt.title("Lollipop Chart of State Frequencies")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    state_freq.plot(kind="scatter", x=None, y=None) if False else None
    plt.figure(figsize=(9,6))
    plt.scatter(state_freq.values, state_freq.index)
    plt.title("Dot Chart of State Frequencies")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.show()

# Compare fatality distributions for a few states, if present.
if {"state", "fatalities"}.issubset(mode_data.columns):
    chosen = mode_data[mode_data["state"].astype(str).isin(["AK", "CA", "NY", "TX", "FL"])]
    if not chosen.empty:
        sns.histplot(data=chosen, x="fatalities", hue="state", bins=15, element="step", stat="count", common_norm=False)
        plt.title("Distribution of Fatalities for Selected States")
        plt.tight_layout()
        plt.show()
