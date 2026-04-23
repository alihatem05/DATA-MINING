import subprocess
import sys

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
    print(f"\n{'='*55}")
    print(f"  Running: {script}")
    print(f"{'='*55}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\nERROR: {script} failed. Stopping.")
        sys.exit(1)

print("\n" + "="*55)
print("  All scripts finished successfully!")
print("  Outputs saved to: outputs/")
print("  Plots saved to:   outputs/plots/")
print("="*55)
