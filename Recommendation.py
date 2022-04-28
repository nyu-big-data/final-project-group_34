#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

    ratings = spark.read.csv(f'hdfs:/user/{netID}', schema='userId INT, movieId INT, rating DOUBLE, timestamp INT') # TODO timestamep type
    ratings.createOrReplaceTempView('ratings')
    avg_scores = spark.sql('SELECT AVG(ratings.rating) FROM ratings GROUP BY ratings.movieId ORDER BY AVG(ratings.rating) DESC')
    avg_scores.show()

    
    # print('Printing boats inferred schema')
    # boats.printSchema()
    # print('Printing sailors inferred schema')
    # sailors.printSchema()
    # # Why does sailors already have a specified schema?

    # print('Reading boats.txt and specifying schema')
    # boats = spark.read.csv('boats.txt', schema='bid INT, bname STRING, color STRING')

    # print('Printing boats with specified schema')
    # boats.printSchema()

    # # Give the dataframe a temporary view so we can run SQL queries
    # boats.createOrReplaceTempView('boats')
    # sailors.createOrReplaceTempView('sailors')
    # # Construct a query
    # print('Example 1: Executing SELECT count(*) FROM boats with SparkSQL')
    # query = spark.sql('SELECT count(*) FROM boats')

    # # Print the results to the console
    # query.show()

    

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('popularity').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
