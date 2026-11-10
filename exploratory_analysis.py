
import os
from pathlib import Path

import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    LongType, IntegerType, DoubleType, StringType
)
from pyspark.sql.functions import count

# ==========================================
# 1. SPARK SESSION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw" / "elliptic"
OUTPUT_DIR = BASE_DIR / "reports" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("Cryptocurrency_EDA")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\n==========================================")
print("CRYPTOCURRENCY EXPLORATORY DATA ANALYSIS")
print("==========================================")

try:

    # ==========================================
    # 2. DEFINE DATASET SCHEMAS
    # ==========================================

    feature_schema = StructType(
        [
            StructField("txId", LongType(), True),
            StructField("time_step", IntegerType(), True)
        ] +
        [
            StructField(f"feature_{i}", DoubleType(), True)
            for i in range(1, 166)
        ]
    )

    class_schema = StructType([
        StructField("txId", LongType(), True),
        StructField("class", StringType(), True)
    ])

    # ==========================================
    # 3. LOAD DATASETS
    # ==========================================

    print("\nLoading datasets...")

    features = (
        spark.read
        .option("header", "false")
        .schema(feature_schema)
        .csv(str(DATA_DIR / "elliptic_txs_features.csv"))
    )

    classes = (
        spark.read
        .option("header", "true")
        .schema(class_schema)
        .csv(str(DATA_DIR / "elliptic_txs_classes.csv"))
    )

    print("Datasets loaded successfully!")

    # ==========================================
    # 4. TOTAL TRANSACTION ANALYSIS
    # ==========================================

    print("\n1. TOTAL TRANSACTION ANALYSIS")

    total_features = features.count()
    total_classes = classes.count()

    print("Total Feature Records:", total_features)
    print("Total Class Records:", total_classes)

    # ==========================================
    # 5. TRANSACTION CLASS DISTRIBUTION
    # ==========================================

    print("\n2. TRANSACTION CLASS DISTRIBUTION")

    class_distribution = (
        classes
        .groupBy("class")
        .agg(count("*").alias("total"))
        .orderBy("class")
    )

    class_distribution.show()

    class_rows = class_distribution.collect()

    class_labels = [row["class"] for row in class_rows]
    class_counts = [row["total"] for row in class_rows]

    plt.figure(figsize=(9, 5))
    plt.bar(class_labels, class_counts)

    plt.title("Cryptocurrency Transaction Class Distribution")
    plt.xlabel("Transaction Class")
    plt.ylabel("Number of Transactions")
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "class_distribution.png")
    plt.close()

    # ==========================================
    # 6. TIME STEP ANALYSIS
    # ==========================================

    print("\n3. TRANSACTION TIME STEP ANALYSIS")

    time_distribution = (
        features
        .groupBy("time_step")
        .agg(count("*").alias("total"))
        .orderBy("time_step")
    )

    time_distribution.show(50)

    time_rows = time_distribution.collect()

    time_steps = [row["time_step"] for row in time_rows]
    time_counts = [row["total"] for row in time_rows]

    plt.figure(figsize=(12, 5))
    plt.plot(time_steps, time_counts, marker="o")

    plt.title("Transactions Across Time Steps")
    plt.xlabel("Time Step")
    plt.ylabel("Number of Transactions")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "time_step_distribution.png")
    plt.close()

    # ==========================================
    # 7. FEATURE STATISTICS
    # ==========================================

    print("\n4. FEATURE STATISTICAL ANALYSIS")

    feature_columns = [
        f"feature_{i}" for i in range(1, 11)
    ]

    feature_statistics = features.select(
        *feature_columns
    ).describe()

    feature_statistics.show(truncate=False)

    # ==========================================
    # 8. CLASS IMBALANCE ANALYSIS
    # ==========================================

    print("\n5. CLASS IMBALANCE ANALYSIS")

    for label, total in zip(class_labels, class_counts):

        percentage = (total / total_classes) * 100

        print(
            f"Class {label}: {total} transactions "
            f"({percentage:.2f}%)"
        )

    # ==========================================
    # 9. FINAL EDA REPORT
    # ==========================================

    print("\n==========================================")
    print("FINAL EXPLORATORY ANALYSIS REPORT")
    print("==========================================")

    print("Total Transactions:", total_features)
    print("Total Class Records:", total_classes)
    print("Number of Time Steps:", len(time_steps))
    print("Number of Features:", len(feature_columns), "analyzed")

    print("\nGenerated Graphs:")

    print("1. class_distribution.png")
    print("2. time_step_distribution.png")

    print("\nOutput Folder:", OUTPUT_DIR)

    print("\nEXPLORATORY DATA ANALYSIS COMPLETED!")

except Exception as e:
    print("\nERROR DURING EXPLORATORY ANALYSIS:")
    print(str(e))
    raise

finally:
    spark.stop()
