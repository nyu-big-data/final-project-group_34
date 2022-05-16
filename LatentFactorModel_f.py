#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as fn
from pyspark.sql import types as T
from pyspark.mllib.evaluation import RankingMetrics
import time

def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''
    maxIters = [50]
    regParams = [0.01]
    ranks = [10]

    rank = 10
    regParam = 0.01
    maxIter = 10

    ratings_train_orig = spark.read.parquet(f'hdfs:/user/{netID}/train_combined_small_set.parquet')
    ratings_val_orig = spark.read.parquet(f'hdfs:/user/{netID}/val_small_set.parquet')


    start_time = time.time()
    print('maxIter: ', maxIter, 'regParam: ', regParam, 'rank: ', rank)
    print('start at: ', start_time)
    ratings_train = ratings_train_orig
    ratings_train.createOrReplaceTempView('ratings_train')
    print("Ratings")
    ratings_train.show()
    als = ALS(rank=rank, maxIter=maxIter, regParam=regParam, userCol='userId', itemCol='movieId', ratingCol='rating', coldStartStrategy="drop")
    model = als.fit(ratings_train)
    ratings_val = ratings_val_orig
    #ratings_val.createOrReplaceTempView('ratings_val')

    predicted = model.recommendForUserSubset(userSubsetRecs, 100)
    def extractMovieIds(rec):
        return [row.movieId for row in rec]

    extractRecMovieIdsUDF = fn.udf(lambda r: extractMovieIds(r), T.ArrayType(T.IntegerType()))
    predicted = predicted.select(
        fn.col('userId').alias('pr_userId'),
        extractRecMovieIdsUDF('recommendations').alias('rec_movie_id_indices')
    )

    print("PREDICTED")
    print(predicted)


    finish_time = time.time()
    print("----- %s seconds -----", finish_time - start_time)

    predicted.write.mode('overwrite').parquet('val_ALS_small_predicted.parquet')




# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    #sc = SparkContext("local", "First App")
    spark = SparkSession.builder.appName('popularity').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
