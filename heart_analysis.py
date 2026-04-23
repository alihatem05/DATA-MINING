import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs/plots", exist_ok=True)

TARGET_COLUMN = "target"
YOUNG_AGE_THRESHOLD = 45

# Load capped (non-normalized) data so plot values are readable
df = pd.read_csv("outputs/processed_data.csv")

# ============================================================
# Heart Disease Distribution - Pie Chart
# ============================================================

counts = df[TARGET_COLUMN].value_counts()
labels = ["No Heart Disease (0)", "Heart Disease (1)"]
sizes = [counts.get(0, 0), counts.get(1, 0)]

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct="%1.1f%%")
plt.title("Heart Disease Distribution")
plt.tight_layout()
plt.savefig("outputs/plots/target_distribution_pie.png", dpi=150)
plt.close()

# ============================================================
# Parameter Study - Feature vs Heart Disease
# ============================================================

# 1. Age vs Heart Disease
plt.figure(figsize=(7, 5))
sns.boxplot(x=TARGET_COLUMN, y="age", data=df)
plt.title("Age vs Heart Disease")
plt.xlabel("Heart Disease (0=No, 1=Yes)")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig("outputs/plots/age_vs_target.png", dpi=150)
plt.close()

# 2. Cholesterol vs Heart Disease
plt.figure(figsize=(7, 5))
sns.boxplot(x=TARGET_COLUMN, y="chol", data=df)
plt.title("Cholesterol vs Heart Disease")
plt.xlabel("Heart Disease (0=No, 1=Yes)")
plt.ylabel("Cholesterol (mg/dl)")
plt.tight_layout()
plt.savefig("outputs/plots/chol_vs_target.png", dpi=150)
plt.close()

# 3. Resting Blood Pressure vs Heart Disease
plt.figure(figsize=(7, 5))
sns.boxplot(x=TARGET_COLUMN, y="trestbps", data=df)
plt.title("Resting Blood Pressure vs Heart Disease")
plt.xlabel("Heart Disease (0=No, 1=Yes)")
plt.ylabel("Resting BP (mm Hg)")
plt.tight_layout()
plt.savefig("outputs/plots/trestbps_vs_target.png", dpi=150)
plt.close()

# 4. Max Heart Rate vs Heart Disease
plt.figure(figsize=(7, 5))
sns.boxplot(x=TARGET_COLUMN, y="thalach", data=df)
plt.title("Max Heart Rate vs Heart Disease")
plt.xlabel("Heart Disease (0=No, 1=Yes)")
plt.ylabel("Max Heart Rate")
plt.tight_layout()
plt.savefig("outputs/plots/thalach_vs_target.png", dpi=150)
plt.close()

# 5. Gender vs Heart Disease
plt.figure(figsize=(7, 5))
sns.countplot(x="sex", hue=TARGET_COLUMN, data=df)
plt.title("Gender vs Heart Disease (0=Female, 1=Male)")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/plots/sex_vs_target.png", dpi=150)
plt.close()

# 6. Chest Pain Type vs Heart Disease
plt.figure(figsize=(8, 5))
sns.countplot(x="cp", hue=TARGET_COLUMN, data=df)
plt.title("Chest Pain Type vs Heart Disease")
plt.xlabel("Chest Pain Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/plots/cp_vs_target.png", dpi=150)
plt.close()

# 7. Exercise Angina vs Heart Disease
plt.figure(figsize=(7, 5))
sns.countplot(x="exang", hue=TARGET_COLUMN, data=df)
plt.title("Exercise-Induced Angina vs Heart Disease")
plt.xlabel("Exercise Angina (0=No, 1=Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/plots/exang_vs_target.png", dpi=150)
plt.close()

# ============================================================
# Observations
# ============================================================

print("\nProject Observations")
print("---------------------------------------------------")

# Younger people analysis
young_df = df[df["age"] < YOUNG_AGE_THRESHOLD]
younger_prone_pct = (young_df[TARGET_COLUMN].sum() / len(young_df)) * 100
print(f"Percentage of younger people (age < {YOUNG_AGE_THRESHOLD}) with heart disease: {younger_prone_pct:.2f}%")

# Gender analysis
sex_rates = df.groupby("sex")[TARGET_COLUMN].mean() * 100
print("\nHeart disease rate by gender (0=Female, 1=Male):")
for sex, rate in sex_rates.items():
    label = "Female" if sex == 0 else "Male"
    print(f"  {label}: {rate:.2f}%")

# Age group analysis
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 40, 50, 60, 100],
    labels=["<=40", "41-50", "51-60", "60+"],
    include_lowest=True
)
age_group_rates = df.groupby("age_group", observed=False)[TARGET_COLUMN].mean() * 100
print("\nHeart disease rate by age group (%):")
for group, rate in age_group_rates.items():
    print(f"  {group}: {rate:.2f}%")

print("\nAnalysis plots saved to outputs/plots/")
