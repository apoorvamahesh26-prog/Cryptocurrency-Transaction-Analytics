
from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)


# -----------------------------------------
# 1. Project paths
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed" / "splits"
REPORT_DIR = BASE_DIR / "reports" / "ml"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------
# 2. Load training dataset
# -----------------------------------------

print("\nLoading training dataset...")

df = pd.read_parquet(DATA_DIR / "train.parquet")

feature_columns = [
    f"feature_{i}" for i in range(1, 166)
]


# -----------------------------------------
# 3. Create validation split
# -----------------------------------------

train_df = df[df["time_step"] <= 27].copy()
validation_df = df[df["time_step"] >= 28].copy()

X_train = train_df[feature_columns]
y_train = train_df["label"].astype(int)

X_val = validation_df[feature_columns]
y_val = validation_df["label"].astype(int)

print("\n========== DATA SPLIT ==========")

print("Model training records:", len(X_train))
print("Validation records:", len(X_val))

print("Training time steps: 1-27")
print("Validation time steps: 28-34")


# -----------------------------------------
# 4. Train Random Forest
# -----------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced_subsample",
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed!")


# -----------------------------------------
# 5. Get validation probabilities
# -----------------------------------------

probabilities = model.predict_proba(X_val)[:, 1]

print("\nValidation PR-AUC:",
      round(average_precision_score(y_val, probabilities), 4))


# -----------------------------------------
# 6. Test different thresholds
# -----------------------------------------

thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]

results = []

print("\n========== THRESHOLD COMPARISON ==========")

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )


# -----------------------------------------
# 7. Save comparison report
# -----------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "threshold_comparison.csv",
    index=False
)

best_result = results_df.loc[
    results_df["F1_Score"].idxmax()
]

print("\n========== BEST VALIDATION THRESHOLD ==========")

print("Threshold:", best_result["Threshold"])
print("Precision:", round(best_result["Precision"], 4))
print("Recall:", round(best_result["Recall"], 4))
print("F1 Score:", round(best_result["F1_Score"], 4))

print("\nReport saved:")
print(REPORT_DIR / "threshold_comparison.csv")

print("\nThreshold tuning completed!")
