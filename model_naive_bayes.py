import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
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

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
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