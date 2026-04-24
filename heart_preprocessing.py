import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

INPUT_PATH = "heart.csv"
OUTPUT_PATH = "outputs/final_data.csv"
PLOTS_DIR = "outputs/plots"
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COLUMN = "target"

os.makedirs(PLOTS_DIR, exist_ok=True)



df = pd.read_csv(INPUT_PATH)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from '{INPUT_PATH}'")



# Describe 
print("\n=== Dataset Summary ===")
print(f"Shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nStatistics:\n{df.describe()}")



# Column types 
categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
numerical_cols = [c for c in df.columns if c not in categorical_cols + [TARGET_COLUMN]]



# Clean 
df = df.drop_duplicates().reset_index(drop=True)

for col in numerical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode().iloc[0])

df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

print(f"Rows after dropping duplicates: {len(df)}")
print(f"Remaining nulls: {df.isnull().sum().sum()}")



# Cap outliers 
for col in numerical_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())

    if n_outliers:
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"{col:<12}  capped {n_outliers} outlier(s)")



# Boxplots 
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"boxplot_{col}.png"), dpi=150)
    plt.close()
print(f"Saved boxplots to {PLOTS_DIR}/")



# Correlation heatmap 
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()
print(f"Saved correlation heatmap to {PLOTS_DIR}/")



# Encode & normalise 
target = df[TARGET_COLUMN]
features = df.drop(columns=[TARGET_COLUMN])

# Save capped (non-normalized) data for analysis plots
df.to_csv("outputs/processed_data.csv", index=False)
print("Saved processed data to 'outputs/processed_data.csv'")

features = pd.get_dummies(features, columns=categorical_cols, drop_first=True)

scaler = StandardScaler()
features[numerical_cols] = scaler.fit_transform(features[numerical_cols])



# Save 
final = pd.concat([features, target], axis=1)
final.to_csv(OUTPUT_PATH, index=False)
print(f"Saved final data to '{OUTPUT_PATH}'  ({final.shape[0]} rows, {final.shape[1]} cols)")