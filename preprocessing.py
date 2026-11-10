
import os
import shutil
from pathlib import Path

# Set Hadoop environment before importing PySpark
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["hadoop_home_dir"] = r"C:\hadoop"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit

import pyarrow as pa
import pyarrow.parquet as pq


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

FEATURES_PATH = BASE_DIR / "data" / "raw" / "elliptic" / "elliptic_txs_features.csv"
CLASSES_PATH = BASE_DIR / "data" / "raw" / "elliptic" / "elliptic_txs_classes.csv"

LABELLED_OUTPUT = BASE_DIR / "data" / "processed" / "labelled_transactions"
UNKNOWN_OUTPUT = BASE_DIR / "data" / "processed" / "unknown_transactions"


# --------------------------------------------------
# 2. Start Spark
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("CryptocurrencyTransactionPreprocessing")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("\nSpark session started successfully.")


# --------------------------------------------------
# 3. Read transaction features
# --------------------------------------------------

print("\nReading transaction features...")

features_raw = (
    spark.read
    .option("header", "false")
    .option("inferSchema", "true")
    .csv(str(FEATURES_PATH))
)

if len(features_raw.columns) != 167:
    raise ValueError(
        f"Expected 167 feature columns, found {len(features_raw.columns)}"
    )

feature_columns = [
    "txId",
    "time_step"
] + [
    f"feature_{i}" for i in range(1, 166)
]

features = features_raw.toDF(*feature_columns)

features = (
    features
    .withColumn("txId", col("txId").cast("long"))
    .withColumn("time_step", col("time_step").cast("integer"))
)

print("Feature dataset loaded.")


# --------------------------------------------------
# 4. Read transaction classifications
# --------------------------------------------------

print("\nReading transaction classifications...")

classes = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .csv(str(CLASSES_PATH))
)

classes = (
    classes
    .withColumn("txId", col("txId").cast("long"))
    .withColumn("class", col("class").cast("string"))
)

print("Classification dataset loaded.")


# --------------------------------------------------
# 5. Join both datasets
# --------------------------------------------------

print("\nJoining transaction features and classes...")

combined = features.join(
    classes,
    on="txId",
    how="left"
)

print("Datasets joined successfully.")


# --------------------------------------------------
# 6. Create numeric labels
# --------------------------------------------------

print("\nCreating transaction labels...")

labelled_data = (
    combined
    .filter(col("class").isin("1", "2"))
    .withColumn(
        "label",
        when(col("class") == "1", 1)
        .otherwise(0)
    )
)

unknown_data = (
    combined
    .filter(
        (col("class") == "unknown") |
        col("class").isNull()
    )
    .withColumn("label", lit(None).cast("integer"))
)


# --------------------------------------------------
# 7. Verify record counts
# --------------------------------------------------

print("\nCalculating record counts...")

combined_count = combined.count()
labelled_count = labelled_data.count()
unknown_count = unknown_data.count()

print("\n========== DATASET SUMMARY ==========")
print("Total transactions:", combined_count)
print("Labelled transactions:", labelled_count)
print("Unknown transactions:", unknown_count)

print("\nLabel meaning:")
print("1 = Illicit")
print("0 = Licit")
print("NULL = Unknown")


# --------------------------------------------------
# 8. Write Parquet using PyArrow
# --------------------------------------------------

def save_as_parquet(dataframe, output_directory, batch_size=2000):

    output_directory = Path(output_directory)

    if output_directory.exists():
        shutil.rmtree(output_directory)

    output_directory.mkdir(parents=True, exist_ok=True)

    output_file = output_directory / "part-00000.parquet"

    writer = None
    batch = []

    try:
        for row in dataframe.toLocalIterator():

            batch.append(row.asDict(recursive=True))

            if len(batch) >= batch_size:

                table = pa.Table.from_pylist(batch)

                if writer is None:
                    writer = pq.ParquetWriter(
                        str(output_file),
                        table.schema,
                        compression="snappy"
                    )

                writer.write_table(table)
                batch.clear()

        if batch:

            table = pa.Table.from_pylist(batch)

            if writer is None:
                writer = pq.ParquetWriter(
                    str(output_file),
                    table.schema,
                    compression="snappy"
                )

            writer.write_table(table)

        if writer is None:
            raise ValueError("No records available to write.")

    finally:
        if writer is not None:
            writer.close()

    print("Saved:", output_file)


# --------------------------------------------------
# 9. Save labelled and unknown datasets
# --------------------------------------------------

print("\nSaving labelled transactions...")

save_as_parquet(
    labelled_data,
    LABELLED_OUTPUT
)

print("\nSaving unknown transactions...")

save_as_parquet(
    unknown_data,
    UNKNOWN_OUTPUT
)


# --------------------------------------------------
# 10. Verify Parquet files
# --------------------------------------------------

print("\n========== PARQUET VERIFICATION ==========")

for output_directory in [LABELLED_OUTPUT, UNKNOWN_OUTPUT]:

    parquet_file = output_directory / "part-00000.parquet"

    parquet_table = pq.read_table(parquet_file)

    print("\nDataset:", output_directory.name)
    print("Records:", parquet_table.num_rows)
    print("Columns:", parquet_table.num_columns)
    print("File size:", round(parquet_file.stat().st_size / (1024 * 1024), 2), "MB")


print("\nPreprocessing completed successfully!")

spark.stop()
