import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import featureNormalize
import sigmoid


dataset = pd.read_csv('./machine-learning-ex2/ex2/ex2data1.txt', header=None)

X_Val = dataset.values[:, 0:2]


x_train = featureNormalize.normalize(X_Val)
m,n = x_train.shape
x_train = np.concatenate((np.ones((m, 1)), x_train), axis=1)
y_train = dataset.values[:, 2:3]

pos = np.where(y_train==1)[0]
neg = np.where(y_train==0)[0]
fig = plt.figure()
plt.ion()
plt.scatter(x_train[pos, 1], x_train[pos, 2], color='red', marker='o')
plt.scatter(x_train[neg, 1], x_train[neg, 2], color='blue', marker='x')
plt.draw()


theta = np.zeros((n+1, 1))


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
theta_Var = tf.Variable(theta, dtype=tf.float32, name='Weights')


h = sigmoid.sigmoid(tf.matmul(x, theta_Var))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)



J = -(1/m)*tf.reduce_sum((y*tf.log(h)+(1-y)*tf.log(1-h)), 0)

print(sess.run(J, {x:x_train, y:y_train}))

optimizer = tf.train.GradientDescentOptimizer(12)

train = optimizer.minimize(J)

for i in range(1000):
    sess.run(train, {x:x_train, y:y_train})

print(sess.run(J, {x:x_train, y:y_train}))



plt.pause(10)







