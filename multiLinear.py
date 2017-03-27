import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import turtle
import featureNormalize

dataset = pd.read_csv('/home/sarthak/PycharmProjects/ML/machine-learning-ex1/ex1/ex1data2.txt', header=None)

dataVal = dataset.values

X_val = dataVal[:, 0:2]

x_train = featureNormalize.normalize(X_val)

m, n = dataVal.shape
theta = np.ones((n,1))


x_train = np.concatenate((np.ones((m, 1)), x_train), axis=1)
y_train = dataVal[:, 2:3]

print(x_train)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
theta_var = tf.Variable(theta, dtype=tf.float32, name='Weights')


h = tf.matmul(x, theta_var)
squared_error = (h-y)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)



J = (1.0/2*m)*tf.reduce_sum(squared_error, 0)


optimizer = tf.train.GradientDescentOptimizer(0.000001)

train = optimizer.minimize(squared_error)

for i in range(100):
    sess.run(train, {x:x_train, y:y_train})
    print(sess.run(J, {x: x_train, y: y_train}))




