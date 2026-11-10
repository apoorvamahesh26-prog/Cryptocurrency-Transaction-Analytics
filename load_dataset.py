
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType
)

# ==========================================
# 1. CREATE SPARK SESSION
# ==========================================

spark = (
    SparkSession.builder
    .appName("Cryptocurrency Transaction Analytics")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\n======================================")
print("CRYPTOCURRENCY DATASET LOADING")
print("======================================")

# ==========================================
# 2. DEFINE DATASET PATHS
# ==========================================

base_path = "data/raw/elliptic/"

features_path = base_path + "elliptic_txs_features.csv"
classes_path = base_path + "elliptic_txs_classes.csv"
edges_path = base_path + "elliptic_txs_edgelist.csv"

# ==========================================
# 3. DEFINE FEATURES SCHEMA
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
# 4. DEFINE CLASSES SCHEMA
# ==========================================

classes_schema = StructType([
    StructField("txId", StringType(), True),
    StructField("class", StringType(), True)
])

# ==========================================
# 5. DEFINE EDGES SCHEMA
# ==========================================

edges_schema = StructType([
    StructField("txId1", StringType(), True),
    StructField("txId2", StringType(), True)
])

# ==========================================
# 6. LOAD FEATURES DATA
# ==========================================

print("\nLoading transaction features...")

features_df = (
    spark.read
    .option("header", "false")
    .option("mode", "PERMISSIVE")
    .schema(features_schema)
    .csv(features_path)
)

# ==========================================
# 7. LOAD CLASSES DATA
# ==========================================

print("Loading transaction classes...")

classes_df = (
    spark.read
    .option("header", "true")
    .schema(classes_schema)
    .csv(classes_path)
)

# ==========================================
# 8. LOAD EDGES DATA
# ==========================================

print("Loading transaction connections...")

edges_df = (
    spark.read
    .option("header", "true")
    .schema(edges_schema)
    .csv(edges_path)
)

# ==========================================
# 9. DISPLAY DATASET COUNTS
# ==========================================

print("\n======================================")
print("DATASET RECORD COUNTS")
print("======================================")

print("Feature Records:", features_df.count())
print("Class Records:", classes_df.count())
print("Edge Records:", edges_df.count())

# ==========================================
# 10. DISPLAY SCHEMAS
# ==========================================

print("\nFEATURES DATASET SCHEMA")
features_df.printSchema()

print("\nCLASSES DATASET SCHEMA")
classes_df.printSchema()

print("\nEDGES DATASET SCHEMA")
edges_df.printSchema()

# ==========================================
# 11. DISPLAY SAMPLE RECORDS
# ==========================================

print("\nSAMPLE TRANSACTION FEATURES")
features_df.select(
    "txId",
    "time_step",
    "feature_1",
    "feature_2",
    "feature_3"
).show(5, truncate=False)

print("\nSAMPLE TRANSACTION CLASSES")
classes_df.show(5, truncate=False)

print("\nSAMPLE TRANSACTION CONNECTIONS")
edges_df.show(5, truncate=False)

# ==========================================
# 12. DISPLAY CLASS DISTRIBUTION
# ==========================================

print("\nTRANSACTION CLASS DISTRIBUTION")

classes_df.groupBy("class").count().orderBy("class").show()

print("\n======================================")
print("DATASET LOADING COMPLETED!")
print("======================================")

spark.stop()
