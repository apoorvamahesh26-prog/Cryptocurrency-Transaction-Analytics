
from pathlib import Path

import pandas as pd

from xgboost import XGBClassifier

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

print("\nLoading training data...")

df = pd.read_parquet(DATA_DIR / "train.parquet")

feature_columns = [
    f"feature_{i}" for i in range(1, 166)
]

X = df[feature_columns]
y = df["label"]
time_steps = df["time_step"]

folds = [
    (17, 22),
    (22, 27),
    (27, 34)
]

configurations = [
    (3, 0.03),
    (3, 0.05),
    (5, 0.03),
    (5, 0.05),
    (7, 0.03),
    (7, 0.05)
]

thresholds = [0.30, 0.50]

results = []

for fold_number, (train_end, val_end) in enumerate(
    folds, start=1
):

    train_mask = time_steps <= train_end

    val_mask = (
        (time_steps > train_end) &
        (time_steps <= val_end)
    )

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_val = X.loc[val_mask]
    y_val = y.loc[val_mask]

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

    print(f"\n========== FOLD {fold_number} ==========")

    for max_depth, learning_rate in configurations:

        print(
            f"\nTesting depth={max_depth}, "
            f"learning_rate={learning_rate}"
        )

        model = XGBClassifier(
            n_estimators=300,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]

        pr_auc = average_precision_score(
            y_val,
            probabilities
        )

        for threshold in thresholds:

            predictions = (
                probabilities >= threshold
            ).astype(int)

            precision = precision_score(
                y_val,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                y_val,
                predictions,
                zero_division=0
            )

            f1 = f1_score(
                y_val,
                predictions,
                zero_division=0
            )

            results.append({
                "Fold": fold_number,
                "Max_Depth": max_depth,
                "Learning_Rate": learning_rate,
                "Threshold": threshold,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "PR_AUC": pr_auc
            })

            print(
                f"Threshold={threshold:.2f} | "
                f"Precision={precision:.4f} | "
                f"Recall={recall:.4f} | "
                f"F1={f1:.4f} | "
                f"PR-AUC={pr_auc:.4f}"
            )

results_df = pd.DataFrame(results)

output_path = (
    REPORT_DIR / "xgboost_tuning_results.csv"
)

results_df.to_csv(output_path, index=False)

summary = (
    results_df
    .groupby(
        ["Max_Depth", "Learning_Rate", "Threshold"]
    )[
        ["Precision", "Recall", "F1", "PR_AUC"]
    ]
    .mean()
    .reset_index()
)

print("\n========== XGBOOST TUNING SUMMARY ==========")

print(
    summary.sort_values(
        "F1", ascending=False
    ).round(4).to_string(index=False)
)

print("\nSaved:", output_path)

print("\nXGBoost hyperparameter tuning completed!")
