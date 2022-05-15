from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics

import pandas as pd

from pyspark import SparkContext
sc = SparkContext("local", "First App")

spark = SparkSession.builder.appName('popularity').getOrCreate()
ratings_train = spark.read.option("header",True).parquet('val_small_set.parquet')

df = ratings_train.toPandas()
df['userId'] = df['userId'].sort_values()
user_movie = list(df.groupby('userId')['movieId'].apply(list))
print("LABEL")
#print(user_movie)

als_rec = spark.read.option("header",True).parquet('val_ALS_small_predicted.parquet')
rec = als_rec.toPandas()
rec['pr_userId'] = rec['pr_userId'].sort_values()
rec = list(rec.groupby('pr_userId')['rec_movie_id_indices'].apply(list))
print('REC')
#print(rec)

inp = list(zip(user_movie, rec))


#for a, b in inp:
# print("A: ", a, "B: ", b)
# print("")

rdd = sc.parallelize(inp)

metrics = RankingMetrics(rdd)

print("Precision 100:", metrics.precisionAt(100))
print("MAP 100:",metrics.meanAveragePrecisionAt(100))

