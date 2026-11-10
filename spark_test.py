
from pyspark.sql import SparkSession

# Create Spark session
spark = (
    SparkSession.builder
    .appName("Cryptocurrency Transaction Analytics")
    .master("local[*]")
    .getOrCreate()
)

print("====================================")
print("CRYPTOCURRENCY ANALYTICS PROJECT")
print("====================================")

print("Spark Version:", spark.version)

# Create sample cryptocurrency transactions
transactions = [
    (1, "Bitcoin", 0.50, 45000),
    (2, "Ethereum", 2.00, 6000),
    (3, "Bitcoin", 1.25, 112500),
    (4, "Ethereum", 3.00, 9000),
    (5, "Bitcoin", 0.10, 9000)
]

columns = [
    "transaction_id",
    "cryptocurrency",
    "quantity",
    "transaction_value"
]

df = spark.createDataFrame(transactions, columns)

print("\nSample Cryptocurrency Transactions:")

df.show()

print("\nTotal Transactions:")
print(df.count())

print("\nTotal Transaction Value:")

df.groupBy("cryptocurrency").sum(
    "transaction_value"
).show()

print("Spark test completed successfully!")

spark.stop()
