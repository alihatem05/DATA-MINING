import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

# ── Load preprocessed data (run heart_preprocessing.py first) ─
df = pd.read_csv("outputs/final_data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# ── Decision Tree ─────────────────────────────────────────────

# 1) Train / Test Split (80% train – 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# 2) Build & Train the Model
clf = DecisionTreeClassifier(max_depth=4, criterion="gini", random_state=42)
clf.fit(X_train, y_train)

# 3) Predictions & Accuracy
y_pred = clf.predict(X_test)

print("\n" + "=" * 45)
print(f"  Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("=" * 45)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall    = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1        = f1_score(y_test, y_pred, average="macro", zero_division=0)
print(f"  Precision (macro) : {precision * 100:.2f}%")
print(f"  Recall    (macro) : {recall * 100:.2f}%")
print(f"  F1-Score  (macro) : {f1 * 100:.2f}%")
print("---------------------------------------------------")

# 4) Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["No Disease", "Disease"],
    cmap="Blues", ax=ax,
)
ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# 5) Tree Visualization
plt.figure(figsize=(22, 10))
plot_tree(
    clf,
    feature_names=X.columns.tolist(),
    class_names=["No Disease", "Disease"],
    filled=True, rounded=True, fontsize=9,
)
plt.title("Decision Tree – Heart Disease (max_depth=4)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# 6) Feature Importances
importances = pd.Series(clf.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=True)

print("\nFeature Importances:")
print(importances.sort_values(ascending=False).to_string())

colors = ["#e74c3c" if v == importances_sorted.max() else "#3498db" for v in importances_sorted]
plt.figure(figsize=(8, 6))
importances_sorted.plot(kind="barh", color=colors)
plt.title("Feature Importances", fontsize=13, fontweight="bold")
plt.xlabel("Gini Importance")
plt.tight_layout()
plt.show()

# 7) Tree Rules (text)
print("\nDecision Tree Rules:")
print(export_text(clf, feature_names=X.columns.tolist()))

# Save predictions for comparison
pd.DataFrame({"actual": y_test.values, "predicted": y_pred}).to_csv(
    "outputs/pred_decision_tree.csv", index=False
)