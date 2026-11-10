
from pathlib import Path
import json

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt


# -----------------------------------------
# 1. Project paths
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed" / "splits"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" / "ml"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.parquet"
TEST_PATH = DATA_DIR / "test.parquet"


# -----------------------------------------
# 2. Load training and testing data
# -----------------------------------------

print("\nLoading training and testing datasets...")

train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

feature_columns = [
    f"feature_{i}" for i in range(1, 166)
]

X_train = train_df[feature_columns]
y_train = train_df["label"].astype(int)

X_test = test_df[feature_columns]
y_test = test_df["label"].astype(int)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of features:", len(feature_columns))


# -----------------------------------------
# 3. Baseline model
# -----------------------------------------

print("\nTraining baseline model...")

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

print("Baseline accuracy:",
      round(accuracy_score(y_test, baseline_predictions), 4))

print("Baseline illicit recall:",
      round(recall_score(y_test, baseline_predictions, zero_division=0), 4))


# -----------------------------------------
# 4. Train Random Forest
# -----------------------------------------

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced_subsample",
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Random Forest training completed!")


# -----------------------------------------
# 5. Make predictions
# -----------------------------------------

print("\nPredicting testing transactions...")

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]


# -----------------------------------------
# 6. Evaluate model
# -----------------------------------------

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

roc_auc = roc_auc_score(y_test, probabilities)

pr_auc = average_precision_score(y_test, probabilities)

print("\n========== RANDOM FOREST RESULTS ==========")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))
print("PR-AUC:", round(pr_auc, 4))


# -----------------------------------------
# 7. Classification report
# -----------------------------------------

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Licit", "Illicit"],
        zero_division=0
    )
)


# -----------------------------------------
# 8. Confusion matrix
# -----------------------------------------

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Licit", "Illicit"]
)

display.plot(values_format="d")
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "random_forest_confusion_matrix.png",
    dpi=300
)

plt.close()


# -----------------------------------------
# 9. Save trained model
# -----------------------------------------

model_path = MODEL_DIR / "random_forest_model.joblib"

joblib.dump(model, model_path)

print("\nModel saved:", model_path)


# -----------------------------------------
# 10. Save evaluation metrics
# -----------------------------------------

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc)
}

metrics_path = REPORT_DIR / "random_forest_metrics.json"

with open(metrics_path, "w") as file:
    json.dump(metrics, file, indent=4)


# -----------------------------------------
# 11. Save feature importance
# -----------------------------------------

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    REPORT_DIR / "random_forest_feature_importance.csv",
    index=False
)

print("\nTop 10 important features:")
print(importance.head(10).to_string(index=False))


# -----------------------------------------
# 12. Final summary
# -----------------------------------------

print("\n========== FINAL SUMMARY ==========")

print("Model: Random Forest")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Features:", len(feature_columns))

print("\nEvaluation metrics saved.")
print("Confusion matrix saved.")
print("Feature importance saved.")

print("\nModel training completed successfully!")
