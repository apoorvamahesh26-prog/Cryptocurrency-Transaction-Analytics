
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Project folders
BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports" / "ml"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Final test-set results at threshold 0.50
results = [
    {
        "Model": "Random Forest",
        "Accuracy": 0.9804,
        "Precision": 0.9749,
        "Recall": 0.7165,
        "F1-score": 0.8260,
        "ROC-AUC": 0.9342,
        "PR-AUC": 0.7968,
        "True Negatives": 15567,
        "False Positives": 20,
        "False Negatives": 307,
        "True Positives": 776
    },
    {
        "Model": "XGBoost",
        "Accuracy": 0.9753,
        "Precision": 0.8636,
        "Recall": 0.7368,
        "F1-score": 0.7952,
        "ROC-AUC": 0.9333,
        "PR-AUC": 0.8006,
        "True Negatives": 15461,
        "False Positives": 126,
        "False Negatives": 285,
        "True Positives": 798
    }
]

df = pd.DataFrame(results)

# Save comparison table
csv_path = REPORT_DIR / "final_model_comparison.csv"
df.to_csv(csv_path, index=False)

print("\n========== FINAL MODEL COMPARISON ==========")
print(df.round(4).to_string(index=False))

# Plot 1: Model metric comparison
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC",
    "PR-AUC"
]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    x - width / 2,
    df.loc[0, metrics].values,
    width,
    label="Random Forest"
)

ax.bar(
    x + width / 2,
    df.loc[1, metrics].values,
    width,
    label="XGBoost"
)

ax.set_title("Random Forest vs XGBoost - Final Test Results")
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.10)
ax.legend()
ax.grid(axis="y", alpha=0.3)

for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)

plt.tight_layout()

metric_chart_path = REPORT_DIR / "final_model_metric_comparison.png"
plt.savefig(metric_chart_path, dpi=300)
plt.close()

# Plot 2: Confusion matrix comparison
confusion_metrics = [
    "True Negatives",
    "False Positives",
    "False Negatives",
    "True Positives"
]

x = np.arange(len(confusion_metrics))

fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(
    x - width / 2,
    df.loc[0, confusion_metrics].values,
    width,
    label="Random Forest"
)

ax.bar(
    x + width / 2,
    df.loc[1, confusion_metrics].values,
    width,
    label="XGBoost"
)

ax.set_title("Confusion Matrix Comparison")
ax.set_ylabel("Number of Transactions")
ax.set_xticks(x)
ax.set_xticklabels(confusion_metrics)
ax.legend()
ax.grid(axis="y", alpha=0.3)

for container in ax.containers:
    ax.bar_label(container, padding=3, fontsize=8)

plt.tight_layout()

confusion_chart_path = REPORT_DIR / "final_confusion_matrix_comparison.png"
plt.savefig(confusion_chart_path, dpi=300)
plt.close()

print("\n========== SAVED FILES ==========")
print("Comparison CSV:", csv_path)
print("Metric chart:", metric_chart_path)
print("Confusion chart:", confusion_chart_path)

print("\nFinal model comparison completed successfully!")
