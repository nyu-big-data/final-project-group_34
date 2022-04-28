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

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def main(spark, netID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    netID : string, netID of student to find files in HDFS
    '''

    # Load the boats.txt and sailors.json data into DataFrame
    ratings = pd.read_csv('hdfs:/user/{netID}/movielens/ml-latest-small/ratings.csv')
    movies = pd.read_csv('hdfs:/user/{netID}/movielens/ml-latest-small/movies.csv')

    user = ratings.userId.unique()

    train_user, testval_user = train_test_split(pd.DataFrame(user), test_size=0.6)
    temp1_movie, temp2_movie = train_test_split(ratings['movieId'], test_size = 0.7)

    train_set = ratings[ratings['userId'].isin(list(train_user[0]))]
    testval_set = ratings[ratings['userId'].isin(list(testval_user[0]))]
    train_sending = testval_set[testval_set['movieId'].isin(list(temp1_movie))]
    testval_cluster = testval_set[testval_set['movieId'].isin(list(temp2_movie))]

    val_user, test_user = train_test_split(testval_cluster['userId'], test_size = 0.5)

    val_set = testval_cluster[testval_cluster['userId'].isin(list(val_user))]
    test_set = testval_cluster[testval_cluster['userId'].isin(list(test_user))]

    print(val_set.head())

    # val_user, test_user = train_test_split(temp_user, test_size = 0.5)

    # temp1_movie, temp2_movie = train_test_split(ratings['movieId'], test_size = 0.7)




    
    
    

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('partitioning').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Call our main routine
    main(spark, netID)
