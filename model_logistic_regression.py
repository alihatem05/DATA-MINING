import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

os.makedirs("outputs/plots", exist_ok=True)

# ============================================================
# Load Data
# ============================================================

df = pd.read_csv("outputs/final_data.csv")

X = df.drop("target", axis=1)
Y = df["target"]

# ============================================================
# Train / Test Split
# ============================================================

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# ============================================================
# SMOTE - Balance Training Data
# ============================================================

sm = SMOTE(random_state=42)
X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train)

print("After SMOTE:")
print(Y_train_res.value_counts())
print("---------------------------------------------------")

# ============================================================
# Logistic Regression
# ============================================================

model = LogisticRegression(max_iter=300, random_state=42)
model.fit(X_train_res, Y_train_res)

pred = model.predict(X_test)

print("\nLogistic Regression")
print(classification_report(Y_test, pred))

accuracy = accuracy_score(Y_test, pred)
error = (1 - accuracy) * 100
precision = precision_score(Y_test, pred, average="macro", zero_division=0)
recall    = recall_score(Y_test, pred, average="macro", zero_division=0)
f1        = f1_score(Y_test, pred, average="macro", zero_division=0)
print(f"Logistic Regression Error %: {error:.2f}%")
print(f"Precision (macro)          : {precision * 100:.2f}%")
print(f"Recall    (macro)          : {recall * 100:.2f}%")
print(f"F1-Score  (macro)          : {f1 * 100:.2f}%")
print("---------------------------------------------------")

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(Y_test, pred), annot=True, fmt="d", cmap="Greens")
plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/plots/cm_logistic_regression.png", dpi=150)
plt.close()

# Save predictions for comparison
pd.DataFrame({"actual": Y_test.values, "predicted": pred}).to_csv(
    "outputs/pred_logistic_regression.csv", index=False
)
