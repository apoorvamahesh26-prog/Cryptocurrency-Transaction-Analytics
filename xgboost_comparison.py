
from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed" / "splits"
REPORT_DIR = BASE_DIR / "reports" / "ml"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading training dataset...")

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

thresholds = [0.30, 0.50]

results = []

for fold_number, (train_end, validation_end) in enumerate(
    folds, start=1
):

    train_mask = time_steps <= train_end

    validation_mask = (
        (time_steps > train_end) &
        (time_steps <= validation_end)
    )

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_val = X.loc[validation_mask]
    y_val = y.loc[validation_mask]

    print(f"\n========== FOLD {fold_number} ==========")

    print("Training records:", len(X_train))
    print("Validation records:", len(X_val))

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

    models = {

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        )
    }

    for model_name, model in models.items():

        print("\nTraining:", model_name)

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
                "Training_Period": f"1-{train_end}",
                "Validation_Period": (
                    f"{train_end + 1}-{validation_end}"
                ),
                "Model": model_name,
                "Threshold": threshold,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "PR_AUC": pr_auc
            })

            print(
                f"Threshold: {threshold:.2f} | "
                f"Precision: {precision:.4f} | "
                f"Recall: {recall:.4f} | "
                f"F1: {f1:.4f} | "
                f"PR-AUC: {pr_auc:.4f}"
            )

results_df = pd.DataFrame(results)

output_path = (
    REPORT_DIR / "xgboost_comparison.csv"
)

results_df.to_csv(output_path, index=False)

print("\n========== AVERAGE MODEL COMPARISON ==========")

summary = (
    results_df
    .groupby(["Model", "Threshold"])[
        ["Precision", "Recall", "F1", "PR_AUC"]
    ]
    .mean()
    .reset_index()
)

print(summary.round(4).to_string(index=False))

print("\nSaved:", output_path)

print("\nXGBoost comparison completed!")
