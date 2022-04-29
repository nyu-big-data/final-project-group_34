#!/usr/bin/env python
# coding: utf-8

# In[5]:

import dask
from sklearn.model_selection import train_test_split




# mov
ratings = pd.read_csv('/scratch/work/courses/DSGA1004-2021/movielens/ml-latest/ratings.csv')






user = pd.DataFrame(ratings.userId.unique())
train_user, testval_user = train_test_split(user, test_size=0.4)


movie = pd.DataFrame(ratings.movieId.unique())
temp1_movie, temp2_movie = train_test_split(movie, test_size = 0.7)
train_set = ratings[ratings['userId'].isin(list(train_user[0]))]
testval_set = ratings[ratings['userId'].isin(list(testval_user[0]))]
train_sending = testval_set[testval_set['movieId'].isin(list(temp1_movie[0]))]
testval_cluster =testval_set[testval_set['movieId'].isin(list(temp2_movie[0]))]

temp = pd.DataFrame(testval_set.userId.unique())
val_user, test_user = train_test_split(temp, test_size = 0.5)
val_set = testval_cluster[testval_cluster['userId'].isin(list(val_user[0]))]
test_set = testval_cluster[testval_cluster['userId'].isin(list(test_user[0]))]






# concatenate back to train set
final_train = train_set.append(train_sending)




# train_small_data = final_train
# val_small_data = val_set
# test_small_data = test_set
train_large_data = final_train
val_large_data = val_set
test_large_data = test_set


# In[31]:


# train_small_data.to_csv("train_small_data.csv")
# val_small_data.to_csv("val_small_data.csv")
# test_small_data.to_csv("test_small_data.csv")
train_large_data.to_csv("train_large_data.csv")
val_large_data.to_csv("val_large_data.csv")
test_large_data.to_csv("test_large_data.csv")


