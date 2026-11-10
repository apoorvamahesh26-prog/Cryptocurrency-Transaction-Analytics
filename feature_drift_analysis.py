
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed" / "splits"
REPORT_DIR = BASE_DIR / "reports" / "ml"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading datasets...")

train_df = pd.read_parquet(DATA_DIR / "train.parquet")
test_df = pd.read_parquet(DATA_DIR / "test.parquet")

early_test = test_df[
    test_df["time_step"].between(35, 42)
].copy()

late_test = test_df[
    test_df["time_step"].between(43, 49)
].copy()

feature_columns = [
    f"feature_{i}" for i in range(1, 166)
]

print("Training records:", len(train_df))
print("Early test records:", len(early_test))
print("Late test records:", len(late_test))

# -------------------------------------------------
# Calculate feature drift
# -------------------------------------------------

def calculate_drift(reference, comparison, description):

    results = []

    for feature in feature_columns:

        ref_values = reference[feature].dropna()
        comp_values = comparison[feature].dropna()

        ref_mean = ref_values.mean()
        comp_mean = comp_values.mean()

        ref_std = ref_values.std()

        # Standardized mean difference using training SD
        if ref_std > 0:
            drift_score = abs(comp_mean - ref_mean) / ref_std
        else:
            drift_score = 0.0

        results.append({
            "Comparison": description,
            "Feature": feature,
            "Reference_Mean": ref_mean,
            "Comparison_Mean": comp_mean,
            "Reference_Std": ref_std,
            "Comparison_Std": comp_values.std(),
            "Drift_Score": drift_score
        })

    return pd.DataFrame(results).sort_values(
        "Drift_Score",
        ascending=False
    )


print("\n========== OVERALL FEATURE DRIFT ==========")

overall_early = calculate_drift(
    train_df, early_test, "Training vs Test 35-42"
)

overall_late = calculate_drift(
    train_df, late_test, "Training vs Test 43-49"
)

print("\nTop 15 changed features: Test 35-42")
print(
    overall_early.head(15)[
        ["Feature", "Drift_Score"]
    ].to_string(index=False)
)

print("\nTop 15 changed features: Test 43-49")
print(
    overall_late.head(15)[
        ["Feature", "Drift_Score"]
    ].to_string(index=False)
)

# -------------------------------------------------
# Compare actual illicit transactions only
# -------------------------------------------------

print("\n========== ILLICIT FEATURE DRIFT ==========")

train_illicit = train_df[
    train_df["label"] == 1
]

early_illicit = early_test[
    early_test["label"] == 1
]

late_illicit = late_test[
    late_test["label"] == 1
]

print("Training illicit records:", len(train_illicit))
print("Early test illicit records:", len(early_illicit))
print("Late test illicit records:", len(late_illicit))

illicit_early = calculate_drift(
    train_illicit,
    early_illicit,
    "Training illicit vs Test illicit 35-42"
)

illicit_late = calculate_drift(
    train_illicit,
    late_illicit,
    "Training illicit vs Test illicit 43-49"
)

print("\nTop 15 illicit feature changes: Test 35-42")
print(
    illicit_early.head(15)[
        ["Feature", "Drift_Score"]
    ].to_string(index=False)
)

print("\nTop 15 illicit feature changes: Test 43-49")
print(
    illicit_late.head(15)[
        ["Feature", "Drift_Score"]
    ].to_string(index=False)
)

# -------------------------------------------------
# Save reports
# -------------------------------------------------

all_results = pd.concat(
    [overall_early, overall_late,
     illicit_early, illicit_late],
    ignore_index=True
)

csv_path = REPORT_DIR / "feature_drift_analysis.csv"

all_results.to_csv(csv_path, index=False)

# -------------------------------------------------
# Create chart
# -------------------------------------------------

top_features = overall_late.head(15).sort_values(
    "Drift_Score"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Drift_Score"]
)

plt.title("Top 15 Feature Changes: Training vs Test 43-49")
plt.xlabel("Absolute Standardized Mean Difference")
plt.ylabel("Feature")
plt.tight_layout()

chart_path = REPORT_DIR / "feature_drift_chart.png"

plt.savefig(chart_path, dpi=300)
plt.close()

print("\n========== SAVED FILES ==========")
print("CSV:", csv_path)
print("Chart:", chart_path)

print("\nFeature drift analysis completed!")
