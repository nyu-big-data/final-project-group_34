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
ratings_train = spark.read.option("header",True).parquet('val_small_set.parquet')

df = ratings_train.toPandas()

user_num = len(df.groupby('userId')['userId'])


user_movie = list(df.groupby('userId')['movieId'].apply(list))

als_rec = spark.read.option("header",True).parquet('val_ALS_small_predicted.parquet')


rec = als_rec.toPandas()

rec_movie = list(rec.groupby('userId')['movieId'].apply(list))



inp = list(zip(user_movie,rec_movie))

rdd = sc.parallelize(inp)

metrics = RankingMetrics(rdd)

print("Precision 100:", metrics.precisionAt(100))
print("MAP 100:",metrics.meanAveragePrecisionAt(100))

