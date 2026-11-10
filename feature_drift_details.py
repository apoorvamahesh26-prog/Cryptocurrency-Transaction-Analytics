
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed" / "splits"
REPORT_DIR = BASE_DIR / "reports" / "ml"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading datasets...")

train_df = pd.read_parquet(DATA_DIR / "train.parquet")
test_df = pd.read_parquet(DATA_DIR / "test.parquet")

early_test = test_df[
    test_df["time_step"].between(35, 42)
]

late_test = test_df[
    test_df["time_step"].between(43, 49)
]

important_features = [
    "feature_101",
    "feature_103",
    "feature_136",
    "feature_100",
    "feature_139",
    "feature_137",
    "feature_2",
    "feature_143",
    "feature_144",
    "feature_109"
]

periods = {
    "Training (1-34)": train_df,
    "Early Test (35-42)": early_test,
    "Late Test (43-49)": late_test
}

results = []

for feature in important_features:

    for period_name, df in periods.items():

        results.append({
            "Feature": feature,
            "Period": period_name,
            "Mean": df[feature].mean(),
            "Std": df[feature].std(),
            "Minimum": df[feature].min(),
            "Maximum": df[feature].max()
        })

result_df = pd.DataFrame(results)

print("\n========== IMPORTANT FEATURE DETAILS ==========")

print(result_df.round(4).to_string(index=False))

output_path = REPORT_DIR / "feature_drift_details.csv"

result_df.to_csv(output_path, index=False)

print("\nSaved:", output_path)

print("\nFeature drift details completed!")
