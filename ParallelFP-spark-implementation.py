from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import concat_ws, col, lit
import time

def mine_and_write_closed_patterns(input_data_path, output_data_path, minimum_support):
    # Initialize a Spark session
    spark = SparkSession.builder.appName("ParallelFPGrowth").getOrCreate()

    # Load data from a text file (adjust the path accordingly)
    df = spark.read.text(input_data_path).toDF("transaction")

    # Split the transaction into an array of items
    df = df.selectExpr("split(transaction, ' ') as items")

    # Create an instance of the FPGrowth algorithm with minSupport set to the provided value
    fp_growth = FPGrowth(itemsCol="items", minSupport=minimum_support, minConfidence=0.5)

    # Record the start time
    start_time = time.time()

    # Fit the model to the data
    model = fp_growth.fit(df)

    # Record the end time
    # end_time = time.time()

    # Mine frequent patterns
    freq_patterns = model.freqItemsets

    # Filter closed frequent patterns (using the 'freq' column and model's minSupport)
    closed_freq_patterns = freq_patterns.filter(freq_patterns.freq >= model.getMinSupport())

    # end_time = time.time()

    # Convert the array of items to a string
    closed_freq_patterns = closed_freq_patterns.withColumn("items_str", concat_ws(' ', col("items"))).drop("items")

    # Convert the 'freq' column to a string
    closed_freq_patterns = closed_freq_patterns.withColumn("freq_str", closed_freq_patterns.freq.cast("string")).drop("freq")

    # Combine the 'freq_str' and 'items_str' columns into a single column
    output_column = concat_ws(' ', col("items_str"), lit("#SUP:"), col("freq_str"))

    # Show the final closed frequent patterns DataFrame
    # print("Final Closed Frequent Patterns:")
    # closed_freq_patterns.select(output_column.alias("output")).show(truncate=False)

    end_time = time.time()

    # Write the closed frequent patterns to a text file (adjust the path accordingly)
    # print(f"Writing output to: {output_data_path}")
    closed_freq_patterns.select(output_column.alias("output")).write.mode("overwrite").text(output_data_path)

    # end_time = time.time()

    # Print statistics
    printStats(minimum_support, len(df.collect()), end_time - start_time)

    # Stop the Spark session
    spark.stop()

def printStats(minSupport, numOfTrans, total_time):
    print("========== Parallel FPGrowth - STATS ============")
    print(f"minSupport : {int(100.0 * minSupport)}%")
    print(f"Total time ~: {total_time} s")
    print("=====================================")

# Example usage
input_data_path = "C:/Users/acer/PycharmProjects/pythonProjectPFP/chess.dat"
output_data_path = "C:/Users/acer/PycharmProjects/pythonProjectPFP/10.1outputfile.dat"
minimum_support = 0.4

mine_and_write_closed_patterns(input_data_path, output_data_path, minimum_support)