import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import featureNormalize

dataset = pd.read_csv('/home/sarthak/PycharmProjects/ML/machine-learning-ex2/ex2/ex2data1.txt', header=None)

X_Val = dataset.values[:, 0:2]



x_train = featureNormalize.normalize(X_Val)
y_train = dataset.values[:, 2:3]




