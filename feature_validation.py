
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# -----------------------------------------
# 1. Project paths
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "labelled_transactions" / "part-00000.parquet"
)

REPORT_DIR = BASE_DIR / "reports" / "ml"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------
# 2. Load labelled dataset
# -----------------------------------------

print("\nLoading labelled transaction dataset...")

df = pq.read_table(DATA_PATH).to_pandas()

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# -----------------------------------------
# 3. Identify feature columns
# -----------------------------------------

feature_columns = [
    f"feature_{i}" for i in range(1, 166)
]

missing_columns = [
    column for column in feature_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing features: {missing_columns}")

features = df[feature_columns]


# -----------------------------------------
# 4. Check missing values
# -----------------------------------------

missing_values = features.isnull().sum()

print("\n========== MISSING VALUE CHECK ==========")
print("Total missing feature values:", missing_values.sum())


# -----------------------------------------
# 5. Check infinite values
# -----------------------------------------

infinite_values = np.isinf(
    features.to_numpy(dtype=np.float64)
).sum()

print("\n========== INFINITE VALUE CHECK ==========")
print("Total infinite feature values:", infinite_values)


# -----------------------------------------
# 6. Check constant features
# -----------------------------------------

constant_features = [
    column for column in feature_columns
    if features[column].nunique(dropna=False) <= 1
]

print("\n========== CONSTANT FEATURE CHECK ==========")
print("Number of constant features:", len(constant_features))

if constant_features:
    print("Constant feature names:", constant_features)


# -----------------------------------------
# 7. Check data types
# -----------------------------------------

non_numeric = [
    column for column in feature_columns
    if not pd.api.types.is_numeric_dtype(features[column])
]

print("\n========== DATA TYPE CHECK ==========")
print("Non-numeric features:", len(non_numeric))

if non_numeric:
    print(non_numeric)


# -----------------------------------------
# 8. Check class distribution by time step
# -----------------------------------------

print("\n========== CLASS DISTRIBUTION BY TIME STEP ==========")

time_distribution = pd.crosstab(
    df["time_step"],
    df["label"]
).rename(columns={
    0: "licit",
    1: "illicit"
})

print(time_distribution.to_string())

time_distribution.to_csv(
    REPORT_DIR / "class_distribution_by_time.csv"
)


# -----------------------------------------
# 9. Save feature validation report
# -----------------------------------------

report = pd.DataFrame({
    "Feature": feature_columns,
    "Data_Type": [
        str(features[column].dtype)
        for column in feature_columns
    ],
    "Missing_Values": [
        int(features[column].isnull().sum())
        for column in feature_columns
    ],
    "Unique_Values": [
        int(features[column].nunique())
        for column in feature_columns
    ]
})

report.to_csv(
    REPORT_DIR / "feature_validation.csv",
    index=False
)


# -----------------------------------------
# 10. Final summary
# -----------------------------------------

print("\n========== FINAL VALIDATION SUMMARY ==========")

print("Total features:", len(feature_columns))
print("Missing values:", int(missing_values.sum()))
print("Infinite values:", int(infinite_values))
print("Constant features:", len(constant_features))
print("Non-numeric features:", len(non_numeric))

print("\nReports saved in:", REPORT_DIR)

print("\nFeature validation completed!")
