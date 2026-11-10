
from pathlib import Path

import pandas as pd
import joblib

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

print("\nLoading datasets...")

train_df = pd.read_parquet(DATA_DIR / "train.parquet")
test_df = pd.read_parquet(DATA_DIR / "test.parquet")

print("Training records:", len(train_df))
print("Testing records:", len(test_df))

print("\nLoading Random Forest model...")

model = joblib.load(
    MODEL_DIR / "random_forest_tuned_model.joblib"
)

feature_columns = [
    f"feature_{i}" for i in range(1, 166)
]

print("\nGenerating test probabilities...")

probabilities = model.predict_proba(
    test_df[feature_columns]
)[:, 1]

test_df = test_df.copy()
test_df["probability"] = probabilities

test_df["prediction_030"] = (
    test_df["probability"] >= 0.30
).astype(int)

test_df["prediction_050"] = (
    test_df["probability"] >= 0.50
).astype(int)

# -------------------------------------------------
# 1. Training and test class distribution
# -------------------------------------------------

print("\n========== CLASS DISTRIBUTION ==========")

for name, dataset in [
    ("Training", train_df),
    ("Testing", test_df)
]:

    counts = dataset["label"].value_counts().sort_index()

    licit = int(counts.get(0, 0))
    illicit = int(counts.get(1, 0))
    total = len(dataset)

    print(f"\n{name} dataset")
    print("Total:", total)
    print("Licit:", licit)
    print("Illicit:", illicit)
    print(
        "Illicit percentage:",
        round((illicit / total) * 100, 2),
        "%"
    )

# -------------------------------------------------
# 2. Compare three chronological periods
# -------------------------------------------------

periods = {
    "Training (1-34)": train_df,
    "Test (35-42)": test_df[
        test_df["time_step"].between(35, 42)
    ],
    "Test (43-49)": test_df[
        test_df["time_step"].between(43, 49)
    ]
}

print("\n========== PERIOD COMPARISON ==========")

period_results = []

for period_name, dataset in periods.items():

    total = len(dataset)
    illicit = int((dataset["label"] == 1).sum())
    licit = int((dataset["label"] == 0).sum())

    period_results.append({
        "Period": period_name,
        "Total": total,
        "Licit": licit,
        "Illicit": illicit,
        "Illicit_Percentage": (
            illicit / total * 100 if total else 0
        )
    })

period_df = pd.DataFrame(period_results)

print(period_df.to_string(index=False))

period_df.to_csv(
    REPORT_DIR / "temporal_period_comparison.csv",
    index=False
)

# -------------------------------------------------
# 3. Prediction probability analysis
# -------------------------------------------------

print("\n========== PREDICTION PROBABILITY ANALYSIS ==========")

probability_results = []

for period_name, dataset in [
    ("Test 35-42", periods["Test (35-42)"]),
    ("Test 43-49", periods["Test (43-49)"])
]:

    print(f"\n{period_name}")

    for actual_label, label_name in [
        (0, "Licit"),
        (1, "Illicit")
    ]:

        subset = dataset[
            dataset["label"] == actual_label
        ]

        if len(subset) == 0:
            continue

        avg_probability = subset["probability"].mean()
        median_probability = subset["probability"].median()

        print(
            f"{label_name} | "
            f"Records: {len(subset)} | "
            f"Average probability: {avg_probability:.4f} | "
            f"Median probability: {median_probability:.4f}"
        )

        probability_results.append({
            "Period": period_name,
            "Actual_Class": label_name,
            "Records": len(subset),
            "Average_Probability": avg_probability,
            "Median_Probability": median_probability
        })

probability_df = pd.DataFrame(probability_results)

probability_df.to_csv(
    REPORT_DIR / "temporal_probability_analysis.csv",
    index=False
)

# -------------------------------------------------
# 4. Overall performance in two test periods
# -------------------------------------------------

print("\n========== TEST PERIOD PERFORMANCE ==========")

for period_name, dataset in [
    ("Time steps 35-42", periods["Test (35-42)"]),
    ("Time steps 43-49", periods["Test (43-49)"])
]:

    y_true = dataset["label"].astype(int)

    print(f"\n{period_name}")

    for threshold, prediction_column in [
        (0.30, "prediction_030"),
        (0.50, "prediction_050")
    ]:

        y_pred = dataset[prediction_column]

        precision = precision_score(
            y_true, y_pred, zero_division=0
        )

        recall = recall_score(
            y_true, y_pred, zero_division=0
        )

        f1 = f1_score(
            y_true, y_pred, zero_division=0
        )

        print(
            f"Threshold: {threshold:.2f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

print("\n========== SAVED FILES ==========")

print(
    REPORT_DIR / "temporal_period_comparison.csv"
)

print(
    REPORT_DIR / "temporal_probability_analysis.csv"
)

print("\nTemporal shift analysis completed!")
