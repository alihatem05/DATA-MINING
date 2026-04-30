import subprocess
import sys
import os

os.environ.setdefault("MPLBACKEND", "Agg")

scripts = [
    "heart_preprocessing.py",
    "model_logistic_regression.py",
    "model_random_forest.py",
    "model_decision_tree.py",
    "model_knn.py",
    "model_naive_bayes.py",
    "heart_analysis.py",
    "model_comparison.py",
]

for script in scripts:
    subprocess.run([sys.executable, script], check=True)
