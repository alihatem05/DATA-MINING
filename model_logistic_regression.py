import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv("outputs/final_data.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(
    [
        ("feature_selection", SelectKBest(score_func=f_classif, k=11)),
        ("classifier", LogisticRegression(max_iter=1000, C=1, solver="lbfgs", random_state=42)),
    ]
)
model.fit(X_train, y_train)
predicted = model.predict(X_test)

cm = confusion_matrix(y_test, predicted)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/plots/cm_logistic_regression.png", dpi=150)
plt.close()

pd.DataFrame({"actual": y_test.values, "predicted": predicted}).to_csv(
    "outputs/pred_logistic_regression.csv", index=False
)
