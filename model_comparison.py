import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

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
    data = pd.read_csv(path)
    actual = data["actual"]
    predicted = data["predicted"]

    acc = accuracy_score(actual, predicted)
    error = (1 - acc) * 100

    compare.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Error %": round(error, 2)
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
