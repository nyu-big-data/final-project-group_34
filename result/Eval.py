from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics

import pandas as pd

from pyspark import SparkContext
sc = SparkContext("local", "First App")

#     for maxIter in maxIters:
#         for regParam in regParams:
#     '''
spark = SparkSession.builder.appName('popularity').getOrCreate()
ratings_train = spark.read.option("header",True).parquet('train_combined_small_set.parquet')

df = ratings_train.toPandas()

user_movie = list(df.groupby('userId')['movieId'].apply(list))

popularity = spark.read.option("header",True).parquet('train_combined_small_set.parquet')

pop = popularity.toPandas()


pop_movie = list(df.groupby('userId')['movieId'].apply(list))
# pop_movie = list(pop['movieId'])

inp = list(zip(user_movie,pop_movie))

rdd = sc.parallelize(inp)

metrics = RankingMetrics(rdd)

print(metrics.meanAveragePrecisionAt(2))

