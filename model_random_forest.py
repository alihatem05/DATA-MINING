import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

os.makedirs("outputs/plots", exist_ok=True)

# ============================================================
# Load Data
# ============================================================

df = pd.read_csv("outputs/encoded_data.csv")

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
# Random Forest
# ============================================================

model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train_res, Y_train_res)

pred = model.predict(X_test)

print("\nRandom Forest")
print(classification_report(Y_test, pred))

accuracy = accuracy_score(Y_test, pred)
error = (1 - accuracy) * 100
print(f"Random Forest Error %: {error:.2f}%")
print("---------------------------------------------------")

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(Y_test, pred), annot=True, fmt="d", cmap="Greens")
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/plots/cm_random_forest.png", dpi=150)
plt.close()

# Feature Importance Plot
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names[indices][:15], importances[indices][:15])
plt.xlabel("Feature Importance")
plt.title("Top Features Affecting Heart Disease (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("outputs/plots/rf_feature_importance.png", dpi=150)
plt.close()

# Save predictions for comparison
pd.DataFrame({"actual": Y_test.values, "predicted": pred}).to_csv(
    "outputs/pred_random_forest.csv", index=False
)
