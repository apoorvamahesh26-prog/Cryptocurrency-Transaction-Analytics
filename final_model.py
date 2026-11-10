from pathlib import Path
import json

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
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
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" / "ml"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.30

print("\nLoading training and test datasets...")

train_df = pd.read_parquet(DATA_DIR / "train.parquet")
test_df = pd.read_parquet(DATA_DIR / "test.parquet")

feature_columns = [
    f"feature_{i}" for i in range(1, 166)
]

X_train = train_df[feature_columns]
y_train = train_df["label"].astype(int)

X_test = test_df[feature_columns]
y_test = test_df["label"].astype(int)

print("\n========== DATASET INFORMATION ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training time steps: 1-34")
print("Testing time steps: 35-49")
print("Selected threshold:", THRESHOLD)

print("\nTraining final Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced_subsample",
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed!")

print("\nEvaluating model on test dataset...")

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
roc_auc = roc_auc_score(y_test, probabilities)
pr_auc = average_precision_score(y_test, probabilities)

print("\n========== FINAL TEST RESULTS ==========")
print(f"Threshold: {THRESHOLD:.2f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")

report = classification_report(
    y_test,
    predictions,
    target_names=["Licit", "Illicit"],
    zero_division=0
)

print(report)

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(y_test, predictions)

print(cm)

# Save final model
model_path = MODEL_DIR / "random_forest_tuned_model.joblib"

joblib.dump(model, model_path)

# Save metrics
metrics = {
    "threshold": THRESHOLD,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "training_records": len(X_train),
    "testing_records": len(X_test),
    "training_time_steps": "1-34",
    "testing_time_steps": "35-49"
}

metrics_path = REPORT_DIR / "final_model_metrics.json"

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

pd.DataFrame(report_dict).transpose().to_csv(
    REPORT_DIR / "final_classification_report.csv"
)

# Save confusion matrix image
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Licit", "Illicit"]
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Final Random Forest Confusion Matrix")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "final_confusion_matrix.png",
    dpi=300
)

plt.close()

print("\n========== SAVED FILES ==========")
print("Model:", model_path)
print("Metrics:", metrics_path)
print("Classification report:",
      REPORT_DIR / "final_classification_report.csv")
print("Confusion matrix:",
      REPORT_DIR / "final_confusion_matrix.png")

print("\nFinal model evaluation completed!")