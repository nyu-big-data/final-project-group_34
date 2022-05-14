#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Use getpass to obtain user netID
import getpass

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as fn
from pyspark.sql import types as T

def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

    maxIters = [5]
    regParams = [0.01]

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
            #test2 = spark.sql('SELECT * FROM ratings_val')
            #test2.show()

            #predicted = model.transform(ratings_val)
            predicted = model.recommendForUserSubset(userSubsetRecs, 100)

            def extractMovieIds(rec):
                return [row.movieId for row in rec]

            extractRecMovieIdsUDF = fn.udf(lambda r: extractMovieIds(r), T.ArrayType(T.IntegerType()))
            predicted = predicted.select(
                fn.col('userId').alias('userId'),
                extractRecMovieIdsUDF('recommendations').alias('rec_movie_id_indices')
            )
            #predicted = predicted.rdd.map(lambda obj: (obj.movieId))
            #print(predicted.take(100))

            #test2 = spark.createDataFrame(predicted, ["userId", "recommendations"])
            #print("TEST2")
            #test2.show()
            #print("PREDICTED")
            #print(predicted)

            label = ratings_val.groupBy("userId").agg(fn.collect_list('movieId').alias('label'))
            #test3 = spark.sql('SELECT * FROM label')
            print("TO LIST")
            label.show()

            combined = predicted.join(label, ['userId'])
            print("COMBINED")
            combined.show()

            

            #predicted.write.mode('overwrite').parquet(f'hdfs:/user/{netID}/val_ALS_small_predicted.parquet')
            # predicted = predicted.na.drop()
            #evaluator = RegressionEvaluator(metricName='rmse', labelCol='label', predictionCol="rec_movie_id_indices")
            #rmse = evaluator.evaluate(predicted)
            #print('maxIter: ', maxIter, 'regParam: ', regParam, 'Root-mean-square error = ' + str(rmse))




# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('popularity').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
