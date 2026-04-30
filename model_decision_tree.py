import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("outputs/final_data.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(
    max_depth=4,
    criterion="entropy",
    min_samples_split=20,
    min_samples_leaf=1,
    random_state=42,
)
model.fit(X_train, y_train)
predicted = model.predict(X_test)

cm = confusion_matrix(y_test, predicted)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/plots/cm_decision_tree.png", dpi=150)
plt.close()

pd.DataFrame({"actual": y_test.values, "predicted": predicted}).to_csv(
    "outputs/pred_decision_tree.csv", index=False
)

plt.figure(figsize=(22, 10))
plot_tree(
    model,
    feature_names=X.columns.tolist(),
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    fontsize=9,
)
plt.title("Decision Tree - Heart Disease (max_depth=4)")
plt.tight_layout()
plt.savefig("outputs/plots/decision_tree.png", dpi=150)
plt.close()

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(8, 6))
importance.plot(kind="barh")
plt.title("Decision Tree Feature Importance")
plt.xlabel("Gini Importance")
plt.tight_layout()
plt.savefig("outputs/plots/decision_tree_feature_importance.png", dpi=150)
plt.close()
