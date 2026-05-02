import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

MODEL_FILES = {
    "Logistic Regression": "outputs/pred_logistic_regression.csv",
    "Random Forest": "outputs/pred_random_forest.csv",
    "Decision Tree": "outputs/pred_decision_tree.csv",
    "KNN": "outputs/pred_knn.csv",
    "Naive Bayes": "outputs/pred_naive_bayes.csv",
}


def calculate_metrics(name, path):
    data = pd.read_csv(path)
    actual, predicted = data["actual"], data["predicted"]
    accuracy = accuracy_score(actual, predicted)
    return {
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision_score(actual, predicted, average="macro", zero_division=0), 4),
        "Recall": round(recall_score(actual, predicted, average="macro", zero_division=0), 4),
        "F1-Score": round(f1_score(actual, predicted, average="macro", zero_division=0), 4),
        "Error %": round((1 - accuracy) * 100, 2),
    }


comparison = [calculate_metrics(name, path) for name, path in MODEL_FILES.items()]
df = pd.DataFrame(comparison).sort_values("Accuracy", ascending=False).reset_index(drop=True)
df.to_csv("outputs/models_comparison.csv", index=False)

print("\nModel Comparison:")
print(df.to_string(index=False))

plt.figure(figsize=(9, 5))
plt.bar(df["Model"], df["Error %"], color="teal")
plt.ylabel("Error Percentage %")
plt.title("Model Error Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/plots/model_error_comparison.png", dpi=150)
plt.close()

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
x = np.arange(len(df))
width = 0.18

fig, ax = plt.subplots(figsize=(12, 6))
for i, metric in enumerate(metrics):
    ax.bar(x + i * width, df[metric], width, label=metric)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(df["Model"], rotation=15, ha="right")
ax.set_ylabel("Score")
ax.set_ylim(0.8, 1.0)
ax.set_title("Model Comparison - Accuracy, Precision, Recall, F1-Score")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/plots/model_metrics_comparison.png", dpi=150)
plt.close()
