import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("preprocessed data.csv")   # replace with your file name

# Features / Target
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Parameter tuning
params = {
    'n_estimators': [200, 500],
    'max_depth': [5, 8, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Train best model
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_

print("Best Parameters:")
print(grid.best_params_)

# Cross Validation Score
cv_score = cross_val_score(best_rf, X, y, cv=5).mean()
print("\nCross Validation Accuracy:", round(cv_score * 100, 2), "%")

# Test Prediction
y_pred = best_rf.predict(X_test)

# Final Accuracy
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", round(acc * 100, 2), "%")

# Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
importance = pd.Series(best_rf.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=importance.values, y=importance.index)
plt.title("Feature Importance")
plt.show()
