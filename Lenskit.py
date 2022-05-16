#!/usr/bin/env python
# coding: utf-8




from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit import topn


# In[6]:


import pandas as pd


# In[7]:


ratings = pd.read_csv('/scratch/work/courses/DSGA1004-2021/movielens/ml-latest-small/ratings.csv')
# ratings = pd.read_csv('/scratch/work/courses/DSGA1004-2021/movielens/ml-latest/ratings.csv')


# In[8]:

# In[9]:


ratings.columns = ratings.columns.str.replace('userId', 'user')
ratings.rename(columns = {'userId':'user', 'movieId':'item'}, inplace = True)




algo_ii = knn.ItemItem(20)
algo_als = als.BiasedMF(50)


# In[12]:


def eval(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = test.user.unique()
    # now we run the recommender
    recs = batch.recommend(fittable, users, 100)
    # add the algorithm name for analyzability
    recs['Algorithm'] = aname
    return recs


# In[ ]:


all_recs = []
test_data = []
for train, test in xf.partition_users(ratings[['user', 'item', 'rating']], 5, xf.SampleFrac(0.2)):
    test_data.append(test)
    all_recs.append(eval('ItemItem', algo_ii, train, test))
    all_recs.append(eval('ALS', algo_als, train, test))




all_recs = pd.concat(all_recs, ignore_index=True)


# In[34]:


test_data = pd.concat(test_data, ignore_index=True)


# In[35]:


rla = topn.RecListAnalysis()
rla.add_metric(topn.ndcg)
results = rla.compute(all_recs, test_data)

# In[36]:


print(results.groupby('Algorithm').ndcg.mean())


# In[37]:


# results.groupby('Algorithm').ndcg.mean().plot.bar()


# In[ ]:




