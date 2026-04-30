import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

X_train = pd.read_csv("outputs/X_train.csv")
X_test = pd.read_csv("outputs/X_test.csv")
y_train = pd.read_csv("outputs/y_train.csv").squeeze()
y_test = pd.read_csv("outputs/y_test.csv").squeeze()

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=7,
    criterion="entropy",
    min_samples_split=20,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)
predicted = model.predict(X_test)

accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted, average="macro", zero_division=0)
recall = recall_score(y_test, predicted, average="macro", zero_division=0)
f1 = f1_score(y_test, predicted, average="macro", zero_division=0)
error = (1 - accuracy) * 100

print("Random Forest:")
print(f"  Accuracy : {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall   : {recall:.4f}")
print(f"  F1-Score : {f1:.4f}")
print(f"  Error %  : {error:.2f}%")

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

importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importance.values, y=importance.index)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("outputs/plots/rf_feature_importance.png", dpi=150)
plt.close()
