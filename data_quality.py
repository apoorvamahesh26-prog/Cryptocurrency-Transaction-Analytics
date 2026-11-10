
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType
)
from pyspark.sql import functions as F

# ==========================================
# 1. CREATE SPARK SESSION
# ==========================================

spark = (
    SparkSession.builder
    .appName("Cryptocurrency Data Quality Analysis")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\n==========================================")
print("CRYPTOCURRENCY DATA QUALITY ANALYSIS")
print("==========================================")

base_path = "data/raw/elliptic/"

# ==========================================
# 2. DEFINE FEATURES SCHEMA
# ==========================================

feature_fields = [
    StructField("txId", StringType(), True),
    StructField("time_step", IntegerType(), True)
]

for i in range(1, 166):
    feature_fields.append(
        StructField(f"feature_{i}", DoubleType(), True)
    )

features_schema = StructType(feature_fields)

# ==========================================
# 3. LOAD ALL THREE DATASETS
# ==========================================

print("\nLoading datasets...")

features = (
    spark.read
    .option("header", "false")
    .schema(features_schema)
    .csv(base_path + "elliptic_txs_features.csv")
)

classes_schema = StructType([
    StructField("txId", StringType(), True),
    StructField("class", StringType(), True)
])

classes = (
    spark.read
    .option("header", "true")
    .schema(classes_schema)
    .csv(base_path + "elliptic_txs_classes.csv")
)

edges_schema = StructType([
    StructField("txId1", StringType(), True),
    StructField("txId2", StringType(), True)
])

edges = (
    spark.read
    .option("header", "true")
    .schema(edges_schema)
    .csv(base_path + "elliptic_txs_edgelist.csv")
)

print("All datasets loaded successfully!")

# ==========================================
# 4. TOTAL RECORD COUNTS
# ==========================================

print("\n1. TOTAL RECORD COUNTS")

feature_count = features.count()
class_count = classes.count()
edge_count = edges.count()

print("Feature Records:", feature_count)
print("Class Records:", class_count)
print("Edge Records:", edge_count)

# ==========================================
# 5. MISSING VALUE ANALYSIS
# ==========================================

print("\n2. MISSING VALUE ANALYSIS")

missing_expressions = [
    F.count(
        F.when(F.col(c).isNull(), 1)
    ).alias(c)
    for c in features.columns
]

missing_result = features.agg(*missing_expressions).first()

total_missing = sum(
    missing_result[c] or 0
    for c in features.columns
)

print("Total Missing Values:", total_missing)

if total_missing == 0:
    print("No missing values found in features.")

print("\nMissing values in classes:")

for column in classes.columns:
    count = classes.filter(
        F.col(column).isNull()
    ).count()
    print(column, ":", count)

print("\nMissing values in edges:")

for column in edges.columns:
    count = edges.filter(
        F.col(column).isNull()
    ).count()
    print(column, ":", count)

# ==========================================
# 6. DUPLICATE TRANSACTION ID ANALYSIS
# ==========================================

print("\n3. DUPLICATE TRANSACTION ID ANALYSIS")

unique_feature_ids = (
    features.select("txId").distinct().count()
)

unique_class_ids = (
    classes.select("txId").distinct().count()
)

duplicate_feature_ids = feature_count - unique_feature_ids
duplicate_class_ids = class_count - unique_class_ids

print("Unique Feature Transaction IDs:", unique_feature_ids)
print("Duplicate Feature Transaction IDs:", duplicate_feature_ids)

print("Unique Class Transaction IDs:", unique_class_ids)
print("Duplicate Class Transaction IDs:", duplicate_class_ids)

# ==========================================
# 7. DUPLICATE EDGE ANALYSIS
# ==========================================

print("\n4. DUPLICATE TRANSACTION CONNECTIONS")

unique_edges = (
    edges.select("txId1", "txId2")
    .distinct()
    .count()
)

duplicate_edges = edge_count - unique_edges

print("Total Connections:", edge_count)
print("Unique Connections:", unique_edges)
print("Duplicate Connections:", duplicate_edges)

# ==========================================
# 8. CLASSIFICATION ANALYSIS
# ==========================================

print("\n5. TRANSACTION CLASSIFICATION")

classes.groupBy("class").count().orderBy("class").show()

valid_classes = ["1", "2", "unknown"]

invalid_classes = classes.filter(
    ~F.col("class").isin(valid_classes) |
    F.col("class").isNull()
).count()

print("Invalid Classification Records:", invalid_classes)

# ==========================================
# 9. TIME STEP ANALYSIS
# ==========================================

print("\n6. TRANSACTION TIME STEP ANALYSIS")

features.groupBy("time_step") \
    .count() \
    .orderBy("time_step") \
    .show(50)

# ==========================================
# 10. FINAL QUALITY REPORT
# ==========================================

print("\n==========================================")
print("FINAL DATA QUALITY REPORT")
print("==========================================")

print("Feature Records:", feature_count)
print("Class Records:", class_count)
print("Edge Records:", edge_count)

print("Missing Feature Values:", total_missing)

print("Duplicate Feature IDs:", duplicate_feature_ids)
print("Duplicate Class IDs:", duplicate_class_ids)
print("Duplicate Edges:", duplicate_edges)

print("Invalid Classifications:", invalid_classes)

print("\nDATA QUALITY ANALYSIS COMPLETED!")

spark.stop()
