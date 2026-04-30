import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")

INPUT_PATH = "heart.csv"
OUTPUT_PATH = "outputs/final_data.csv"
PLOTS_DIR = "outputs/plots"
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COLUMN = "target"

os.makedirs("outputs", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# Load data
df = pd.read_csv(INPUT_PATH)


# Column types
categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
numerical_cols = [c for c in df.columns if c not in categorical_cols + [TARGET_COLUMN]]


# Clean data
df = df.drop_duplicates().reset_index(drop=True)
df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode().iloc[0])


# Detect outliers
outlier_rows = []
for col in numerical_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_rows.append([col, q1, q3, iqr, lower, upper, count])

pd.DataFrame(
    outlier_rows,
    columns=["feature", "q1", "q3", "iqr", "lower_bound", "upper_bound", "outlier_count"],
).to_csv("outputs/outlier_summary.csv", index=False)


# Boxplots
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"boxplot_{col}.png"), dpi=150)
    plt.close()


# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()


# Save cleaned data for analysis plots (before encoding)
df.to_csv("outputs/processed_data.csv", index=False)

target = df[TARGET_COLUMN]
features = df.drop(columns=[TARGET_COLUMN])

for col in categorical_cols:
    drop_first = features[col].nunique() == 2
    dummies = pd.get_dummies(features[col], columns=[col], prefix=col, drop_first=drop_first).astype(int)
    features = pd.concat([features.drop(columns=[col]), dummies], axis=1)


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

X_train.to_csv("outputs/X_train.csv", index=False)
X_test.to_csv("outputs/X_test.csv", index=False)
y_train.to_csv("outputs/y_train.csv", index=False)
y_test.to_csv("outputs/y_test.csv", index=False)


# Save full encoded dataset (unscaled) for backward-compatible analysis
final = pd.concat([features, target], axis=1)
final.to_csv(OUTPUT_PATH, index=False)
