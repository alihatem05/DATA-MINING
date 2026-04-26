import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

os.makedirs("outputs/plots", exist_ok=True)


# =========================
# Load data (NO PRINTS)
# =========================
df = pd.read_csv("outputs/final_data.csv")


# =========================
# Split data
# =========================
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# Train model
# =========================
model = GaussianNB()
model.fit(X_train, y_train)


# =========================
# Predict
# =========================
y_pred = model.predict(X_test)


# =========================
# ONLY YOUR OUTPUT
# =========================

accuracy  = accuracy_score(y_test, y_pred)
error     = (1 - accuracy) * 100
precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall    = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1        = f1_score(y_test, y_pred, average="macro", zero_division=0)
cm        = confusion_matrix(y_test, y_pred)

print("\nNaive Bayes")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))
print(f"Accuracy          : {accuracy * 100:.2f}%")
print(f"Error %           : {error:.2f}%")
print(f"Precision (macro) : {precision * 100:.2f}%")
print(f"Recall    (macro) : {recall * 100:.2f}%")
print(f"F1-Score  (macro) : {f1 * 100:.2f}%")
print("\nConfusion Matrix:")
print(cm)


# =========================
# Confusion Matrix Plot
# =========================
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/plots/cm_naive_bayes.png", dpi=150)
plt.close()

# Save predictions for comparison
pd.DataFrame({"actual": y_test.values, "predicted": y_pred}).to_csv(
    "outputs/pred_naive_bayes.csv", index=False
)