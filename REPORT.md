# Checkpoint Report

### 1. Partitioning
File: Partition_process.py

### 2. Baseline Model
File: BaselineModel.py

Result for small dataset
- train, validation, test dataset results in order
![alt text](https://github.com/nyu-big-data/final-project-group_34/blob/main/result/images/popularity_small.png)
Precision 100: 0.000683760683760684 \
MAP 100: 8.763292737651712e-05

Result for large dataset
- train, validation, test dataset results in order
![alt text](https://github.com/nyu-big-data/final-project-group_34/blob/main/result/images/large_popularity.png)
Precision 100: 3.639937393076843e-07 \
MAP 100: 8.211891037205615e-09

### 3. Latent Factor Model
File: LatentFactorModel.py

Result for small dataset
```
maxIter:  5 regParam:  0.01 Root-mean-square error = 1.292935382204839
maxIter:  5 regParam:  0.1 Root-mean-square error = 0.9134382532601439
maxIter:  10 regParam:  0.01 Root-mean-square error = 1.2921431266279613
maxIter:  10 regParam:  0.1 Root-mean-square error = 0.9133044520773148

maxIter: 5, regParam: 0.01
Precision 100: 0.07572649572649572
MAP 100: 0.06845067505040847
```


### 4. Evaluation
File: Eval.py - Baseline Model \
      Eval_ALS.py - LatentFactorModel


