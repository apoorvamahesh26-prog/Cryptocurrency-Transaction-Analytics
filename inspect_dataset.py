
import csv
import os

DATA_FOLDER = "data/raw/elliptic"

files = [
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv"
]

print("=" * 65)
print("CRYPTOCURRENCY DATASET INSPECTION")
print("=" * 65)

for filename in files:

    filepath = os.path.join(DATA_FOLDER, filename)

    print(f"\nFile: {filename}")

    if not os.path.exists(filepath):
        print("ERROR: File not found!")
        continue

    size_mb = os.path.getsize(filepath) / (1024 * 1024)

    print(f"File Size: {size_mb:.2f} MB")

    with open(filepath, "r", newline="", encoding="utf-8") as file:

        reader = csv.reader(file)

        for row_number, row in enumerate(reader):

            if row_number == 0:
                print("Number of Columns:", len(row))
                print("First Row:", row[:10])

            elif row_number == 1:
                print("Second Row:", row[:10])
                break

print("\nDataset inspection completed!")
