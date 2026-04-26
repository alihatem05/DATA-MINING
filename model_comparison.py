import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.makedirs("outputs/plots", exist_ok=True)

# ============================================================
# Load predictions saved by each model file
# ============================================================

model_files = {
    "Logistic Regression": "outputs/pred_logistic_regression.csv",
    "Random Forest":        "outputs/pred_random_forest.csv",
    "Decision Tree":        "outputs/pred_decision_tree.csv",
    "KNN":                  "outputs/pred_knn.csv",
    "Naive Bayes":          "outputs/pred_naive_bayes.csv",
}

compare = []

for name, path in model_files.items():
    data      = pd.read_csv(path)
    actual    = data["actual"]
    predicted = data["predicted"]

    acc       = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted, average="macro", zero_division=0)
    recall    = recall_score(actual, predicted, average="macro", zero_division=0)
    f1        = f1_score(actual, predicted, average="macro", zero_division=0)
    error     = (1 - acc) * 100

    compare.append({
        "Model":     name,
        "Accuracy":  round(acc, 4),
        "Precision": round(precision, 4),
        "Recall":    round(recall, 4),
        "F1-Score":  round(f1, 4),
        "Error %":   round(error, 2),
    })

df_compare = pd.DataFrame(compare)
print("\nModel Comparison:")
print(df_compare.to_string(index=False))

df_compare.to_csv("outputs/models_comparison.csv", index=False)

# ============================================================
# Error Comparison Bar Chart
# ============================================================

plt.figure(figsize=(9, 5))
plt.bar(df_compare["Model"], df_compare["Error %"], color="teal")
plt.ylabel("Error Percentage %")
plt.title("Model Error Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/plots/model_error_comparison.png", dpi=150)
plt.close()

# ============================================================
# Multi-Metric Grouped Bar Chart
# ============================================================

metrics   = ["Accuracy", "Precision", "Recall", "F1-Score"]
x         = np.arange(len(df_compare["Model"]))
bar_width = 0.18
colors    = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6"]

fig, ax = plt.subplots(figsize=(12, 6))
for i, (metric, color) in enumerate(zip(metrics, colors)):
    ax.bar(x + i * bar_width, df_compare[metric], bar_width, label=metric, color=color)

ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(df_compare["Model"], rotation=15, ha="right")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.set_title("Model Comparison – Accuracy, Precision, Recall, F1-Score")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/plots/model_metrics_comparison.png", dpi=150)
plt.close()
