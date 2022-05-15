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

    # ranks = [10, 50, 100, 200]
    # maxIters = [10, 50, 100]
    # regParams = [0.01, 0.1, 1]
    # alphas = [1, 40]

    ranks = [10]
    maxIters = [10]
    regParams = [0.01]
    alphas = [1]

    for rank in ranks:
        for maxIter in maxIters:
            for regParam in regParams:
                for alpha in alphas:
                    ratings_train = spark.read.parquet(f'hdfs:/user/{netID}/train_combined_small_set.parquet')
                    ratings_train.createOrReplaceTempView('ratings_train')

                    als = ALS(rank=rank, maxIter=maxIter, regParam=regParam, alpha=alpha, \
                              userCol='userId', itemCol='movieId', ratingCol='rating', coldStartStrategy="drop")
                    model = als.fit(ratings_train)
                    #predicted = predicted.na.drop()

                    ratings_val = spark.read.parquet(f'hdfs:/user/{netID}/val_small_set.parquet')  # TODO timestamep type
                    predicted = model.transform(ratings_val)
                    evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol="prediction")
                    rmse = evaluator.evaluate(predicted)
                    print('rank: ', rank, 'maxIter: ', maxIter, 'regParam: ', regParam, 'alpha: ', alpha, 'Root-mean-square error = ' + str(rmse))


# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('ALSforRMSE').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
