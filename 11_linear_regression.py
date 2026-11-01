# Assignment 10

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("C:\\Users\\akrit\\Desktop\\EDA_courseProject_23BDS0038\\USSeatBelts.csv")

# Handle missing values
for col in df.columns:
    if df[col].isna().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicate records
print("Duplicate records:", df.duplicated().sum())
df = df.drop_duplicates()

# Select independent and dependent variables
X = df[["miles"]]
Y = df["fatalities"]

# Descriptive statistics
print("\nDESCRIPTIVE STATISTICS")
print(df[["miles", "fatalities"]].describe())

print("\nMean:\n", df[["miles", "fatalities"]].mean())
print("\nMedian:\n", df[["miles", "fatalities"]].median())
print("\nVariance:\n", df[["miles", "fatalities"]].var())
print("\nStandard Deviation:\n", df[["miles", "fatalities"]].std())

# Histograms
plt.figure(figsize=(7, 4))
plt.hist(X["miles"], bins=30, color="skyblue", edgecolor="black")
plt.title("Histogram of Miles")
plt.xlabel("Miles")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(7, 4))
plt.hist(Y, bins=30, color="orange", edgecolor="black")
plt.title("Histogram of Fatalities")
plt.xlabel("Fatalities")
plt.ylabel("Frequency")
plt.show()

# Density plots
sns.kdeplot(x=df["miles"], fill=True)
plt.title("Density Plot of Miles")
plt.show()

sns.kdeplot(x=df["fatalities"], fill=True)
plt.title("Density Plot of Fatalities")
plt.show()

# Boxplots
plt.figure(figsize=(6, 4))
sns.boxplot(y=df["miles"])
plt.title("Boxplot of Miles")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(y=df["fatalities"])
plt.title("Boxplot of Fatalities")
plt.show()

# Outlier detection using IQR
for col in ["miles", "fatalities"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"\nOutliers in {col}: {len(outliers)}")
    print("Lower limit:", lower)
    print("Upper limit:", upper)

# Scatter plot
plt.figure(figsize=(7, 4))
plt.scatter(X, Y, color="blue", alpha=0.6)
plt.title("Miles vs Fatalities")
plt.xlabel("Miles")
plt.ylabel("Fatalities")
plt.show()

# Correlation
correlation = df["miles"].corr(df["fatalities"])
print("\nPearson Correlation:", correlation)

# Regression line
plt.figure(figsize=(7, 4))
sns.regplot(
    data=df,
    x="miles",
    y="fatalities",
    line_kws={"color": "red"}
)
plt.title("Regression Line: Miles vs Fatalities")
plt.show()

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=123
)

# Simple Linear Regression
model = LinearRegression()
model.fit(X_train, Y_train)

# Model predictions
Y_pred = model.predict(X_test)

# Intercept and slope
print("\nINTERCEPT:", model.intercept_)
print("SLOPE:", model.coef_[0])

# Regression equation
print(
    f"\nRegression Equation: Y = "
    f"{model.intercept_:.6f} + "
    f"{model.coef_[0]:.10f} * X"
)

# R-squared
print("\nR-SQUARED:", r2_score(Y_test, Y_pred))

# Residuals
residuals = Y_test - Y_pred

print("\nRESIDUALS:")
print(residuals.head())

# Evaluation metrics
MAE = mean_absolute_error(Y_test, Y_pred)
MSE = mean_squared_error(Y_test, Y_pred)
RMSE = np.sqrt(MSE)

nonzero = Y_test != 0
MAPE = np.mean(
    np.abs(residuals[nonzero] / Y_test[nonzero])
) * 100

print("\nMODEL EVALUATION")
print("MAE:", MAE)
print("MSE:", MSE)
print("RMSE:", RMSE)
print("MAPE:", MAPE)