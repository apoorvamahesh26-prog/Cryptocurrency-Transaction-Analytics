
from pathlib import Path

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed" / "splits"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" / "ml"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading test dataset...")

df = pd.read_parquet(DATA_DIR / "test.parquet")

print("Test records:", len(df))

print("\nLoading trained Random Forest model...")

model = joblib.load(
    MODEL_DIR / "random_forest_tuned_model.joblib"
)

feature_columns = [
    f"feature_{i}" for i in range(1, 166)
]

X_test = df[feature_columns]
y_test = df["label"].astype(int)

print("\nGenerating predictions...")

probabilities = model.predict_proba(X_test)[:, 1]

df["probability"] = probabilities

results = []

print("\n========== TIME-WISE PERFORMANCE ==========")

for time_step in sorted(df["time_step"].unique()):

    step_df = df[df["time_step"] == time_step]

    actual = step_df["label"].astype(int)

    step_probabilities = step_df["probability"]

    for threshold in [0.30, 0.50]:

        predicted = (
            step_probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            actual, predicted, zero_division=0
        )

        recall = recall_score(
            actual, predicted, zero_division=0
        )

        f1 = f1_score(
            actual, predicted, zero_division=0
        )

        tp = int(((actual == 1) & (predicted == 1)).sum())
        fp = int(((actual == 0) & (predicted == 1)).sum())
        fn = int(((actual == 1) & (predicted == 0)).sum())

        results.append({
            "Time_Step": int(time_step),
            "Threshold": threshold,
            "Total_Transactions": len(step_df),
            "Actual_Illicit": int(actual.sum()),
            "True_Positive": tp,
            "False_Positive": fp,
            "False_Negative": fn,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1
        })

        print(
            f"Time: {time_step} | "
            f"Threshold: {threshold:.2f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

results_df = pd.DataFrame(results)

output_path = REPORT_DIR / "timewise_performance.csv"

results_df.to_csv(output_path, index=False)

# Prepare F1 comparison chart

plt.figure(figsize=(12, 6))

for threshold in [0.30, 0.50]:

    subset = results_df[
        results_df["Threshold"] == threshold
    ]

    plt.plot(
        subset["Time_Step"],
        subset["F1_Score"],
        marker="o",
        label=f"Threshold {threshold:.2f}"
    )

plt.title("Random Forest Time-wise F1 Score")
plt.xlabel("Time Step")
plt.ylabel("F1 Score")
plt.ylim(0, 1.05)
plt.xticks(sorted(df["time_step"].unique()))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

chart_path = REPORT_DIR / "timewise_f1_comparison.png"

plt.savefig(chart_path, dpi=300)
plt.close()

print("\n========== SAVED FILES ==========")
print("CSV:", output_path)
print("Chart:", chart_path)

print("\nTime-wise performance analysis completed!")
