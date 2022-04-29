#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

    ratings_train = spark.read.parquet(f'hdfs:/user/{netID}/train_small_pq.parquet')
    ratings_train.createOrReplaceTempView('ratings_train')
    test1 = spark.sql('SELECT * FROM ratings_train')
    test1.show()

    #ratings_train = spark.read.csv(f'hdfs:/user/{netID}/train_small_data.csv', schema='userId INT, movieId INT, rating DOUBLE, timestamp INT') # TODO timestamep type
    #ratings_train_RDD = spark.createDataFrame(ratings_train)
    #ratings_train.createOrReplaceTempView('ratings_train')

    #score = spark.sql('SELECT * FROM ratings_train WHERE userId=null')
    #score.show()


    als = ALS(maxIter=5, regParam=0.01, userCol='userId', itemCol='movieId', ratingCol='rating', coldStartStrategy="nan")
    model = als.fit(ratings_train)

    ratings_test = spark.read.parquet(f'hdfs:/user/{netID}/test_small_pq.parquet') # TODO timestamep type
    ratings_test.createOrReplaceTempView('ratings_test')
    test2 = spark.sql('SELECT * FROM ratings_test')
    test2.show()    

    predicted = model.transform(ratings_test)
    predicted = predicted.na.drop()
    evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol="prediction")
    rmse = evaluator.evaluate(predicted)
    print("Root-mean-square error = " + str(rmse))




# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('popularity').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
