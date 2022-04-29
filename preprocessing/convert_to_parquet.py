#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Import command line arguments and helper functions(if necessary)
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession


def main(spark):
    '''Main routine for run for Storage optimization template.
    Parameters
    ----------
    spark : SparkSession object

    '''

    # change to parquet for Part 2.4
    train_small_pq = spark.read.csv('hdfs:/user/yl7143/train_small_data.csv', header=True,
                                     schema='id INT, userId INT, movieId INT, rating DOUBLE, timestamp INT')
    train_small_pq.createOrReplaceTempView('train_small_pq')
    train_small_pq.write.mode('overwrite').parquet('hdfs:/user/yl7143/train_small_pq.parquet')

    test_small_pq = spark.read.csv('hdfs:/user/yl7143/test_small_data.csv', header=True,
                                      schema='id INT, userId INT, movieId INT, rating DOUBLE, timestamp INT')
    test_small_pq.createOrReplaceTempView('test_small_pq')
    test_small_pq.write.mode('overwrite').parquet('hdfs:/user/yl7143/test_small_pq.parquet')




# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('convert').getOrCreate()

    # If you wish to command line arguments, look into the sys library(primarily sys.argv)
    # Details are here: https://docs.python.org/3/library/sys.html
    # If using command line arguments, be sure to add them to main function

    main(spark)

