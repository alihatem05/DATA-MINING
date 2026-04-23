import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COLUMN = "target"


def load_data(path: str = "heart.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def describe_dataset(df: pd.DataFrame) -> dict:
    categorical_columns = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
    numerical_columns = [
        col for col in df.columns if col not in categorical_columns + [TARGET_COLUMN]
    ]

    print("\nCategorical columns:")
    print(categorical_columns)
    print("\nNumerical columns:")
    print(numerical_columns)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nStatistics:")
    print(df.describe())

    return {
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
    }


def clean_data(df: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    numerical_columns = [
        col for col in cleaned.columns if col not in categorical_columns + [TARGET_COLUMN]
    ]

    for col in numerical_columns:
        if cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    for col in categorical_columns:
        if cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode().iloc[0])

    if cleaned[TARGET_COLUMN].dtype != "int64":
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)

    print("\nAfter cleaning:")
    print(cleaned.isnull().sum())
    print(f"Rows after dropping duplicates: {len(cleaned)}")

    return cleaned


def outlier_statistics(df: pd.DataFrame, numerical_columns: list[str]) -> dict:
    quartiles = {}

    for col in numerical_columns:
        q1 = df[col].quantile(0.25)
        q2 = df[col].quantile(0.50)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers_count = ((df[col] < lower) | (df[col] > upper)).sum()

        quartiles[col] = {
            "Q1": q1,
            "Q2": q2,
            "Q3": q3,
            "IQR": iqr,
            "Lower Bound": lower,
            "Upper Bound": upper,
            "Outliers Count": int(outliers_count),
        }

        print(f"===== {col} =====")
        for key, value in quartiles[col].items():
            print(f"{key}: {value}")
        print()

    return quartiles


def cap_outliers(
    df: pd.DataFrame, numerical_columns: list[str], quartiles: dict
) -> pd.DataFrame:
    capped = df.copy()

    for col in numerical_columns:
        if quartiles[col]["Outliers Count"] == 0:
            continue

        lower = quartiles[col]["Lower Bound"]
        upper = quartiles[col]["Upper Bound"]
        capped[col] = capped[col].clip(lower=lower, upper=upper)

    print("\nHead after capping:")
    print(capped.head())
    return capped


def plot_boxplots(df: pd.DataFrame, numerical_columns: list[str]) -> None:
    os.makedirs("outputs/plots", exist_ok=True)
    for col in numerical_columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(f"outputs/plots/boxplot_{col}.png", dpi=150)
        plt.close()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    os.makedirs("outputs/plots", exist_ok=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("outputs/plots/correlation_heatmap.png", dpi=150)
    plt.close()


def encode_features(df: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=categorical_columns, drop_first=False)


def normalize_features(df: pd.DataFrame, numerical_columns: list[str]) -> pd.DataFrame:
    normalized = df.copy()
    scaler = MinMaxScaler()
    normalized[numerical_columns] = scaler.fit_transform(normalized[numerical_columns])
    return normalized


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    df = load_data()

    info = describe_dataset(df)
    categorical_columns = info["categorical_columns"]
    numerical_columns = info["numerical_columns"]

    df_clean = clean_data(df, categorical_columns)

    quartiles = outlier_statistics(df_clean, numerical_columns)
    df_capped = cap_outliers(df_clean, numerical_columns, quartiles)

    plot_boxplots(df_clean, numerical_columns)
    print("Saved: outputs/plots/boxplot_*.png")

    plot_correlation_heatmap(df_capped)
    print("Saved: outputs/plots/correlation_heatmap.png")

    df_capped.to_csv("outputs/processed_data.csv", index=False)
    print("\nSaved: outputs/processed_data.csv")

    df_encoded = encode_features(df_capped, categorical_columns)

    encoded_numerical = [col for col in df_encoded.columns if col not in [TARGET_COLUMN]
                         and df_encoded[col].dtype in ["float64", "int64"]
                         and col in numerical_columns]
    df_normalized = normalize_features(df_encoded, encoded_numerical)

    df_normalized.to_csv("outputs/encoded_data.csv", index=False)
    print("Saved: outputs/encoded_data.csv")
