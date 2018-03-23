import tensorflow as tf
import pandas as pd
import numpy as np


#read in file with labels and file names
data = pd.read_csv('db/labels.csv', delimiter = ',')



# #split in train and test
split = np.random.choice( data.shape[0] , size = 10, replace = False)
print(split)
print(data[1,1])
#train = data[split,]
#test = data[-split,]
# 
# train_primary = train[train.temp == 1,]
# train_not_primary = train[train.temp == 0,]
