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
print("\nMISSING VALUES BY COLUMN\n", df.isna().sum())
print("\nTOTAL MISSING VALUES:", df.isna().sum().sum())
print("\nROWS WITH MISSING VALUES\n", df[df.isna().any(axis=1)].head(10))
clean_rows = df.dropna()
print("\nAFTER DROPPING ROWS WITH MISSING VALUES:", clean_rows.shape)

# Separate examples, each starts from the original data.
mean_imputed = df.copy()
if "income" in mean_imputed:
    mean_imputed["income"] = mean_imputed["income"].fillna(mean_imputed["income"].mean())
median_imputed = df.copy()
if "age" in median_imputed:
    median_imputed["age"] = median_imputed["age"].fillna(median_imputed["age"].median())
mode_imputed = df.copy()
if "seatbelt" in mode_imputed and not mode_imputed["seatbelt"].mode().empty:
    mode_imputed["seatbelt"] = mode_imputed["seatbelt"].fillna(mode_imputed["seatbelt"].mode()[0])
print("\nMEAN-IMPUTED INCOME MISSING:", mean_imputed.get("income", pd.Series(dtype=float)).isna().sum())
print("MEDIAN-IMPUTED AGE MISSING:", median_imputed.get("age", pd.Series(dtype=float)).isna().sum())
print("MODE-IMPUTED SEATBELT MISSING:", mode_imputed.get("seatbelt", pd.Series(dtype=float)).isna().sum())

ffill = df.ffill()
bfill = df.bfill()
print("\nFORWARD-FILL SAMPLE\n", ffill.head())
print("\nBACKWARD-FILL SAMPLE\n", bfill.head())
filled = mode_imputed.copy()
filled = filled.fillna({c: "Unknown" for c in filled.select_dtypes(exclude=np.number).columns})
print("\nDUPLICATE ROWS:", filled.duplicated().sum())
filled = filled.drop_duplicates()
print("SHAPE AFTER DEDUPLICATION:", filled.shape)

if "fatalities" in filled.columns:
    filled = filled.rename(columns={"fatalities":"fatality_rate"})
if {"miles","fatality_rate"}.issubset(filled.columns):
    filled["miles_per_fatality"] = filled["miles"] / filled["fatality_rate"].replace(0, np.nan)
print("\nDERIVED DATA SAMPLE\n", filled.head())
if "income" in df:
    x = pd.to_numeric(df["income"], errors="coerce").dropna()
    print("\nINCOME MIN/MAX/RANGE:", x.min(), x.max(), (x.min(), x.max()))
    print("INCOME VARIANCE:", x.var(), "SD:", x.std(), "IQR:", x.quantile(.75)-x.quantile(.25))
