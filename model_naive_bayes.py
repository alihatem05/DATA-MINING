import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.preprocessing._discretization")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing._discretization")

CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

df = pd.read_csv("heart.csv").drop_duplicates().reset_index(drop=True)
X = df.drop("target", axis=1)
y = df["target"]
numeric_columns = [c for c in X.columns if c not in CATEGORICAL_COLUMNS]

preprocessor = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("bins", KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile", subsample=None)),
                ]
            ),
            numeric_columns,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]
            ),
            CATEGORICAL_COLUMNS,
        ),
    ]
)

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", CategoricalNB(alpha=0.5, min_categories=1)),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
predicted = model.predict(X_test)

cm = confusion_matrix(y_test, predicted)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/plots/cm_naive_bayes.png", dpi=150)
plt.close()

pd.DataFrame({"actual": y_test.values, "predicted": predicted}).to_csv(
    "outputs/pred_naive_bayes.csv", index=False
)
