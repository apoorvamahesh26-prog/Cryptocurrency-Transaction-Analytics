
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------
# 1. Project paths
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "labelled_transactions"
    / "part-00000.parquet"
)

REPORT_DIR = BASE_DIR / "reports" / "ml"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------
# 2. Load labelled dataset
# -----------------------------------------

print("\nLoading labelled transactions...")

table = pq.read_table(DATA_PATH, columns=["label"])
df = table.to_pandas()

print("Dataset loaded successfully.")


# -----------------------------------------
# 3. Calculate class distribution
# -----------------------------------------

class_counts = df["label"].value_counts().sort_index()

class_names = {
    0: "Licit",
    1: "Illicit"
}

distribution = pd.DataFrame({
    "Class": [class_names[int(i)] for i in class_counts.index],
    "Transactions": class_counts.values,
    "Percentage": (
        class_counts.values / len(df) * 100
    ).round(2)
})


# -----------------------------------------
# 4. Display results
# -----------------------------------------

print("\n========== CLASS DISTRIBUTION ==========")

print(distribution.to_string(index=False))

print("\nTotal labelled transactions:", len(df))

majority_count = class_counts.max()
minority_count = class_counts.min()

imbalance_ratio = majority_count / minority_count

print("Majority class transactions:", majority_count)
print("Minority class transactions:", minority_count)

print("Class imbalance ratio: 1 :",
      round(imbalance_ratio, 2))


# -----------------------------------------
# 5. Create bar chart
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    distribution["Class"],
    distribution["Transactions"]
)

plt.title("Licit vs Illicit Transaction Distribution")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")

plt.tight_layout()

chart_path = REPORT_DIR / "class_distribution.png"

plt.savefig(chart_path, dpi=300)
plt.show()

print("\nChart saved successfully:")
print(chart_path)


# -----------------------------------------
# 6. Save distribution report
# -----------------------------------------

csv_path = REPORT_DIR / "class_distribution.csv"

distribution.to_csv(csv_path, index=False)

print("\nCSV report saved:")
print(csv_path)

print("\nClass distribution analysis completed!")
