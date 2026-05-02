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

print("--- 1. Load ---")
df = pd.read_csv(INPUT_PATH)
print(f"  {df.shape[0]} rows, {df.shape[1]} columns")

categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
numerical_cols = [c for c in df.columns if c not in categorical_cols + [TARGET_COLUMN]]

print("\n--- 2. Clean ---")
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"  Duplicates removed: {before - len(df)}  |  rows remaining: {len(df)}")

df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

num_filled = 0
for col in numerical_cols:
    missing = df[col].isna().sum()
    if missing:
        df[col] = df[col].fillna(df[col].median())
        print(f"  '{col}': {missing} missing → filled with median")
        num_filled += missing
    else:
        df[col] = df[col].fillna(df[col].median())

cat_filled = 0
for col in categorical_cols:
    missing = df[col].isna().sum()
    if missing:
        mode_val = df[col].mode().iloc[0]
        df[col] = df[col].fillna(mode_val)
        print(f"  '{col}': {missing} missing → filled with mode ({mode_val})")
        cat_filled += missing
    else:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

if num_filled == 0 and cat_filled == 0:
    print("  No missing values.")

print("\n--- 3. Outliers (IQR) ---")
outlier_rows = []
for col in numerical_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_rows.append([col, q1, q3, iqr, lower, upper, count])
    flag = " <<<" if count > 0 else ""
    print(f"  {col:<12}  bounds=[{lower:.3g}, {upper:.3g}]  outliers={count}{flag}")

pd.DataFrame(
    outlier_rows,
    columns=["feature", "q1", "q3", "iqr", "lower_bound", "upper_bound", "outlier_count"],
).to_csv("outputs/outlier_summary.csv", index=False)

print("\n--- 4. Plots ---")
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"boxplot_{col}.png"), dpi=150)
    plt.close()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()
print(f"  Saved {len(numerical_cols)} boxplots + correlation heatmap → {PLOTS_DIR}/")

df.to_csv("outputs/processed_data.csv", index=False)

print("\n--- 5. Encoding ---")
target = df[TARGET_COLUMN]
features = df.drop(columns=[TARGET_COLUMN])

for col in categorical_cols:
    unique_vals = features[col].nunique()
    drop_first = unique_vals == 2
    dummies = pd.get_dummies(features[col], columns=[col], prefix=col, drop_first=drop_first).astype(int)
    features = pd.concat([features.drop(columns=[col]), dummies], axis=1)
print(f"  One-hot encoded {len(categorical_cols)} columns → {features.shape[1]} total features")

print("\n--- 6. Split (80/20) ---")
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

X_train.to_csv("outputs/X_train.csv", index=False)
X_test.to_csv("outputs/X_test.csv", index=False)
y_train.to_csv("outputs/y_train.csv", index=False)
y_test.to_csv("outputs/y_test.csv", index=False)

print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

final = pd.concat([features, target], axis=1)
final.to_csv(OUTPUT_PATH, index=False)
print("\nPreprocessing complete.")
