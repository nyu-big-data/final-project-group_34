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

def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

    ranks = [10, 50, 100, 200]
    maxIters = [10, 50, 100]
    regParams = [0.01, 0.1, 1]

    for rank in ranks:
        for maxIter in maxIters:
            for regParam in regParams:
                ratings_train = spark.read.parquet(f'hdfs:/user/{netID}/train_combined_small_set.parquet')
                ratings_train.createOrReplaceTempView('ratings_train')

                als = ALS(rank = rank, maxIter=maxIter, regParam=regParam, \
                          userCol='userId', itemCol='movieId', ratingCol='rating', coldStartStrategy="drop")
                model = als.fit(ratings_train)

                ratings_val = spark.read.parquet(f'hdfs:/user/{netID}/val_small_set.parquet') # TODO timestamep type
                #ratings_val.createOrReplaceTempView('ratings_val')
                userSubsetRecs = ratings_val.select("userId").distinct().sort("userId")
                #userSubsetRecs.show()
                #predicted = model.transform(ratings_val)
                predicted = model.recommendForUserSubset(userSubsetRecs, 100)

                def extractMovieIds(rec):
                    return [row.movieId for row in rec]

                extractRecMovieIdsUDF = fn.udf(lambda r: extractMovieIds(r), T.ArrayType(T.IntegerType()))
                predicted = predicted.select(fn.col('userId').alias('pr_userId'),
                                             extractRecMovieIdsUDF('recommendations').alias('rec_movie_id_indices'))
                print("PREDICTED")
                print(predicted)

                label = ratings_val.groupBy("userId").agg(fn.collect_list('movieId').alias('label'))
                #print("TO LIST")
                #label.show()

                # combined = predicted.join(fn.broadcast(label), fn.col('pr_userId') == fn.col('userId'))\
                #     .rdd.map(lambda r: (r.rec_movie_id_indices, r.label))

                # metrics = RankingMetrics(combined)
                # print('rank: ', rank, 'maxIter: ', maxIter, 'regParam: ', regParam, 'alpha: ', alpha)
                # print('PrecisionAtK: ', metrics.precisionAt(100))
                # print('MAP: ', metrics.meanAveragePrecision)

                predicted.write.mode('overwrite').parquet(f'hdfs:///user/yl7143/val_ALS_small_predicted_{rank}_{maxIter}_{regParam}.parquet')
                # predicted = predicted.na.drop()
                #evaluator = RegressionEvaluator(metricName='rmse', labelCol='label', predictionCol="rec_movie_id_indices")
                #rmse = evaluator.evaluate(predicted)




# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    #sc = SparkContext("local", "First App")
    spark = SparkSession.builder.appName('popularity').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
