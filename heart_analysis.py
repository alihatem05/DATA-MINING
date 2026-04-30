import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TARGET = "target"
PLOTS_DIR = "outputs/plots"
YOUNG_AGE_THRESHOLD = 45

df = pd.read_csv("outputs/processed_data.csv")


def save_boxplot(feature, title, ylabel, filename):
    plt.figure(figsize=(7, 5))
    sns.boxplot(x=TARGET, y=feature, data=df)
    plt.title(title)
    plt.xlabel("Heart Disease (0=No, 1=Yes)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=150)
    plt.close()


def save_countplot(feature, title, xlabel, filename):
    plt.figure(figsize=(7, 5))
    sns.countplot(x=feature, hue=TARGET, data=df)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=150)
    plt.close()


counts = df[TARGET].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(
    [counts.get(0, 0), counts.get(1, 0)],
    labels=["No Heart Disease (0)", "Heart Disease (1)"],
    autopct="%1.1f%%",
)
plt.title("Heart Disease Distribution")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/target_distribution_pie.png", dpi=150)
plt.close()

boxplots = [
    ("age", "Age vs Heart Disease", "Age", "age_vs_target.png"),
    ("chol", "Cholesterol vs Heart Disease", "Cholesterol (mg/dl)", "chol_vs_target.png"),
    ("trestbps", "Resting Blood Pressure vs Heart Disease", "Resting BP (mm Hg)", "trestbps_vs_target.png"),
    ("thalach", "Max Heart Rate vs Heart Disease", "Max Heart Rate", "thalach_vs_target.png"),
]
for args in boxplots:
    save_boxplot(*args)

countplots = [
    ("sex", "Gender vs Heart Disease (0=Female, 1=Male)", "Sex", "sex_vs_target.png"),
    ("cp", "Chest Pain Type vs Heart Disease", "Chest Pain Type", "cp_vs_target.png"),
    ("exang", "Exercise-Induced Angina vs Heart Disease", "Exercise Angina (0=No, 1=Yes)", "exang_vs_target.png"),
]
for args in countplots:
    save_countplot(*args)

young_df = df[df["age"] < YOUNG_AGE_THRESHOLD]
young_rate = young_df[TARGET].mean() * 100
gender_rates = df.groupby("sex")[TARGET].mean() * 100

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 40, 50, 60, 100],
    labels=["<=40", "41-50", "51-60", "60+"],
    include_lowest=True,
)
age_group_rates = df.groupby("age_group", observed=False)[TARGET].mean() * 100

observations = pd.DataFrame(
    [
        {"section": "young_age", "group": f"age < {YOUNG_AGE_THRESHOLD}", "heart_disease_rate": young_rate},
        {"section": "gender", "group": "Female", "heart_disease_rate": gender_rates.get(0, 0)},
        {"section": "gender", "group": "Male", "heart_disease_rate": gender_rates.get(1, 0)},
        *[
            {"section": "age_group", "group": str(group), "heart_disease_rate": rate}
            for group, rate in age_group_rates.items()
        ],
    ]
)
observations.to_csv("outputs/project_observations.csv", index=False)
