import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("outputs/final_data.csv")

# Features & target
X = df.drop("target", axis=1)
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("KNN Full Comparison (Metric + Weights)\n")

results = []

for metric in ["manhattan", "euclidean"]:
    for weight in ["uniform", "distance"]:

        print(f"\n===== Metric: {metric} | Weight: {weight} =====")

        best_acc = 0
        best_k = 0

        for k in range(1, 12, 2):
            knn = KNeighborsClassifier(
                n_neighbors=k,
                weights=weight,
                metric=metric
            )

            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            print(f"K = {k} -> Accuracy = {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_k = k

        print(f"Best K = {best_k}")
        print(f"Best Accuracy = {best_acc:.4f}")

        results.append((metric, weight, best_k, best_acc))

best_overall = max(results, key=lambda x: x[3])

print("\n==========================")
print(f"Best Metric = {best_overall[0]}")
print(f"Best Weight = {best_overall[1]}")
print(f"Best K = {best_overall[2]}")
print(f"Best Accuracy = {best_overall[3]:.4f}")

best_metric, best_weight, best_k, _ = best_overall

final_model = KNeighborsClassifier(
    n_neighbors=best_k,
    weights=best_weight,
    metric=best_metric
)

final_model.fit(X_train, y_train)
y_final_pred = final_model.predict(X_test)

cm = confusion_matrix(y_test, y_final_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nConfusion Matrix (labeled):")
print("TN =", cm[0][0], "| FP =", cm[0][1])
print("FN =", cm[1][0], "| TP =", cm[1][1])