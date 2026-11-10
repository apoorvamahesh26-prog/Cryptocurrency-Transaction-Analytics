
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split


# -----------------------------------------
# 1. Project paths
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = (
    BASE_DIR / "data" / "processed"
    / "labelled_transactions" / "part-00000.parquet"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------
# 2. Load dataset
# -----------------------------------------

print("\nLoading labelled transactions...")

df = pq.read_table(INPUT_PATH).to_pandas()

print("Total records:", len(df))


# -----------------------------------------
# 3. Chronological split
# -----------------------------------------

# Use time steps 1-34 for training
# Use time steps 35-49 for testing

train_df = df[df["time_step"] <= 34].copy()
test_df = df[df["time_step"] >= 35].copy()

print("\n========== CHRONOLOGICAL SPLIT ==========")

print("Training records:", len(train_df))
print("Testing records:", len(test_df))

print("Training time steps:",
      train_df["time_step"].min(),
      "to",
      train_df["time_step"].max())

print("Testing time steps:",
      test_df["time_step"].min(),
      "to",
      test_df["time_step"].max())


# -----------------------------------------
# 4. Check class distribution
# -----------------------------------------

print("\nTraining class distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nTesting class distribution:")
print(test_df["label"].value_counts().sort_index())


# -----------------------------------------
# 5. Separate features and labels
# -----------------------------------------

feature_columns = [
    f"feature_{i}" for i in range(1, 166)
]

X_train = train_df[feature_columns]
y_train = train_df["label"]

X_test = test_df[feature_columns]
y_test = test_df["label"]


# -----------------------------------------
# 6. Save split datasets
# -----------------------------------------

train_df.to_parquet(
    OUTPUT_DIR / "train.parquet",
    index=False
)

test_df.to_parquet(
    OUTPUT_DIR / "test.parquet",
    index=False
)

print("\nSaved:")
print(OUTPUT_DIR / "train.parquet")
print(OUTPUT_DIR / "test.parquet")


# -----------------------------------------
# 7. Final verification
# -----------------------------------------

print("\n========== FINAL SUMMARY ==========")

print("Total labelled records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nData splitting completed!")
