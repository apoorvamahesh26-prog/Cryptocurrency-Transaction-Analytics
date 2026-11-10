
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed" / "splits"
REPORT_DIR = BASE_DIR / "reports" / "ml"
MODEL_DIR = BASE_DIR / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.50

print("\nLoading training and testing data...")

train_df = pd.read_parquet(DATA_DIR / "train.parquet")
test_df = pd.read_parquet(DATA_DIR / "test.parquet")

feature_columns = [
    f"feature_{i}" for i in range(1, 166)
]

X_train = train_df[feature_columns]
y_train = train_df["label"]

X_test = test_df[feature_columns]
y_test = test_df["label"]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nTraining XGBoost model...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=7,
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

model.fit(X_train, y_train)

print("Model training completed!")

print("\nGenerating predictions...")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= THRESHOLD
).astype(int)

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

roc_auc = roc_auc_score(
    y_test, probabilities
)

pr_auc = average_precision_score(
    y_test, probabilities
)

print("\n========== FINAL XGBOOST RESULTS ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")

report = classification_report(
    y_test,
    predictions,
    target_names=["Licit", "Illicit"],
    zero_division=0
)

print(report)

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

print(cm)

# Save model
model_path = MODEL_DIR / "xgboost_final_model.joblib"

joblib.dump(model, model_path)

# Save metrics
metrics = {
    "model": "XGBoost",
    "max_depth": 7,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "threshold": THRESHOLD,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "training_records": len(X_train),
    "testing_records": len(X_test)
}

metrics_path = REPORT_DIR / "final_xgboost_metrics.json"

with open(metrics_path, "w") as file:
    json.dump(metrics, file, indent=4)

# Save classification report
report_dict = classification_report(
    y_test,
    predictions,
    target_names=["Licit", "Illicit"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()

report_path = (
    REPORT_DIR / "final_xgboost_classification_report.csv"
)

report_df.to_csv(report_path)

# Save confusion matrix chart
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Licit", "Illicit"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Final XGBoost Confusion Matrix")
plt.tight_layout()

chart_path = REPORT_DIR / "final_xgboost_confusion_matrix.png"

plt.savefig(chart_path, dpi=300)
plt.close()

print("\n========== SAVED FILES ==========")

print("Model:", model_path)
print("Metrics:", metrics_path)
print("Classification report:", report_path)
print("Confusion matrix:", chart_path)

print("\nFinal XGBoost evaluation completed!")
