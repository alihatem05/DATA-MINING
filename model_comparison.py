import os
import pandas as pd
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
    data = pd.read_csv(path)
    actual = data["actual"]
    predicted = data["predicted"]

    acc = accuracy_score(actual, predicted)
    prec = precision_score(actual, predicted, zero_division=0)
    rec = recall_score(actual, predicted, zero_division=0)
    f1 = f1_score(actual, predicted, zero_division=0)
    error = (1 - acc) * 100

    compare.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "Error %": round(error, 2)
    })

df_compare = pd.DataFrame(compare)
print("\nModel Comparison:")
print(df_compare.to_string(index=False))

best_recall = df_compare.loc[df_compare["Recall"].idxmax()]
best_f1 = df_compare.loc[df_compare["F1 Score"].idxmax()]

print("------------------------------------------------------------------")
print(f"Best Model by Recall:   {best_recall['Model']}  ({best_recall['Recall']:.4f})")
print(f"Best Model by F1 Score: {best_f1['Model']}  ({best_f1['F1 Score']:.4f})")

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
# Performance Heatmap (Precision / Recall / F1)
# ============================================================

heatmap_df = df_compare.set_index("Model")[["Precision", "Recall", "F1 Score"]]

plt.figure(figsize=(8, 5))
sns.heatmap(heatmap_df, annot=True, cmap="Greens", vmin=0, vmax=1)
plt.title("Model Performance on Heart Disease (class=1)")
plt.tight_layout()
plt.savefig("outputs/plots/model_performance_heatmap.png", dpi=150)
plt.close()

print("\nComparison plots saved to outputs/plots/")
