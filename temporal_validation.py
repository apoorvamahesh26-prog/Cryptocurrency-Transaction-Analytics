
from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed" / "splits"
REPORT_DIR = BASE_DIR / "reports" / "ml"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading training dataset...")

df = pd.read_parquet(DATA_DIR / "train.parquet")

feature_columns = [
    f"feature_{i}" for i in range(1, 166)
]

# Three chronological validation windows
validation_windows = [
    {"train_end": 17, "val_start": 18, "val_end": 22},
    {"train_end": 22, "val_start": 23, "val_end": 27},
    {"train_end": 27, "val_start": 28, "val_end": 34}
]

results = []

for fold_number, window in enumerate(validation_windows, start=1):

    train_end = window["train_end"]
    val_start = window["val_start"]
    val_end = window["val_end"]

    train_df = df[df["time_step"] <= train_end].copy()

    val_df = df[
        (df["time_step"] >= val_start) &
        (df["time_step"] <= val_end)
    ].copy()

    X_train = train_df[feature_columns]
    y_train = train_df["label"].astype(int)

    X_val = val_df[feature_columns]
    y_val = val_df["label"].astype(int)

    print(f"\n========== FOLD {fold_number} ==========")
    print(f"Training time steps: 1-{train_end}")
    print(f"Validation time steps: {val_start}-{val_end}")
    print("Training records:", len(train_df))
    print("Validation records:", len(val_df))

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    print("\nThreshold comparison:")

    for threshold in [0.30, 0.50]:

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

        pr_auc = average_precision_score(
            y_val, probabilities
        )

        print(
            f"Threshold: {threshold:.2f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f} | "
            f"PR-AUC: {pr_auc:.4f}"
        )

        results.append({
            "Fold": fold_number,
            "Training_Period": f"1-{train_end}",
            "Validation_Period": f"{val_start}-{val_end}",
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "PR_AUC": pr_auc
        })

results_df = pd.DataFrame(results)

output_path = REPORT_DIR / "temporal_validation_results.csv"

results_df.to_csv(output_path, index=False)

print("\n========== VALIDATION SUMMARY ==========")

print(
    results_df[
        ["Fold", "Validation_Period", "Threshold",
         "Precision", "Recall", "F1_Score", "PR_AUC"]
    ].to_string(index=False)
)

print("\nResults saved to:")
print(output_path)

print("\nTemporal validation completed!")
