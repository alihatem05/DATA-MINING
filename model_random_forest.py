import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv("outputs/final_data.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=1,
    max_features=None,
    class_weight=None,
    random_state=42,
)
model.fit(X_train, y_train)
predicted = model.predict(X_test)

cm = confusion_matrix(y_test, predicted)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/plots/cm_random_forest.png", dpi=150)
plt.close()

pd.DataFrame({"actual": y_test.values, "predicted": predicted}).to_csv(
    "outputs/pred_random_forest.csv", index=False
)

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importance.values, y=importance.index)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("outputs/plots/rf_feature_importance.png", dpi=150)
plt.close()
