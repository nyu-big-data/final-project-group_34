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

    maxIters = [5, 10]
    regParams = [0.01, 0.1]

    for maxIter in maxIters:
        for regParam in regParams:

            ratings_train = spark.read.parquet(f'hdfs:/user/{netID}/train_combined_small_set.parquet')
            ratings_train.createOrReplaceTempView('ratings_train')
            # test1 = spark.sql('SELECT * FROM ratings_train')
            # test1.show()

            # ratings_train = spark.read.csv(f'hdfs:/user/{netID}/train_small_data.csv', schema='userId INT, movieId INT, rating DOUBLE, timestamp INT') # TODO timestamep type
            # ratings_train_RDD = spark.createDataFrame(ratings_train)
            # ratings_train.createOrReplaceTempView('ratings_train')
            # score = spark.sql('SELECT * FROM ratings_train WHERE userId=null')
            # score.show()

            als = ALS(maxIter=maxIter, regParam=regParam, userCol='userId', itemCol='movieId', ratingCol='rating', coldStartStrategy="drop")
            model = als.fit(ratings_train)

            ratings_val = spark.read.parquet(f'hdfs:/user/{netID}/val_small_set.parquet') # TODO timestamep type
            ratings_val.createOrReplaceTempView('ratings_val')
            userSubsetRecs = ratings_val.select(als.getUserCol()).distinct()
            #test2 = spark.sql('SELECT * FROM ratings_test')
            #test2.show()

            #predicted = model.transform(ratings_val)
            #print(predicted)
            predicted = model.recommendForUserSubset(userSubsetRecs, 100)
            print(predicted)

            #predicted.write.mode('overwrite').parquet(f'hdfs:/user/{netID}/val_ALS_small_predicted.parquet')
            # predicted = predicted.na.drop()
            evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol="prediction")
            rmse = evaluator.evaluate(predicted)
            print('maxIter: ', maxIter, 'regParam: ', regParam, 'Root-mean-square error = ' + str(rmse))




# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('popularity').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
