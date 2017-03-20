import tensorflow as tf
import numpy as np
import pandas as pd


dataset = pd.read_csv('/home/sarthak/PycharmProjects/ML/machine-learning-ex1/ex1/ex1data1.txt', header=None)

val = dataset.values

x_train = val[:, 0:1]
m, n = x_train.shape



x_train = np.concatenate((np.ones((m, 1)), x_train), axis=1)
print x_train**2
y_train = val[:, 1:2]
theta = np.zeros((n, 1))



x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

h = (1/2*m)*(x*theta-y)**2

sess = tf.Session()

print sess.run(h, {x:x_train, y:y_train})








