#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from lightfm import LightFM


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    ratings_train = spark.read.parquet('/home/yl7143/final-project-group_34/result/train_combined_small_set.parquet')
    ratings_train.createOrReplaceTempView('ratings_train')

    model = LightFM(loss='warp')
    model.fit(ratings_train, epochs=30, num_threads=8, verbose=True)

    ratings_val = spark.read.parquet('/home/yl7143/final-project-group_34/result/val_small_set.parquet') # TODO timestamep type
    ratings_val.createOrReplaceTempView('ratings_val')

    # Evaluate the trained model
    test_precision = precision_at_k(model, ratings_val, k=100).mean()
    print("test_precision: ", test_precision)




# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('improvedALS').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
