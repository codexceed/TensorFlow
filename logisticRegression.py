import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import featureNormalize


dataset = pd.read_csv('./machine-learning-ex2/ex2/ex2data1.txt', header=None)

X_Val = dataset.values[:, 0:2]

print(X_Val)


x_train = featureNormalize.normalize(X_Val)
m,n = x_train.shape
np.concatenate((np.ones((m, 1)), x_train), axis=1)
y_train = dataset.values[:, 2:3]
theta = np.zeros((n+1, 1))
print(x_train)
print(y_train)









