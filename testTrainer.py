import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import turtle

#retrieve data from file
dataset = pd.read_csv('/home/sarthak/PycharmProjects/ML/machine-learning-ex1/ex1/ex1data1.txt', header=None)

#import data into nd.array
val = dataset.values

#split data into required training data variables
x_train = val[:, 0:1]
m = x_train.shape[0] #number of data samples
y_train = val[:, 1:2]
fig = plt.figure()
plt.scatter(x_train,y_train, color='red', marker='x')
plt.ion()
plt.show()
theta = np.zeros((2, 1)) #initialize theta to 0

#declare tensorflow varibles to be trained
theta1 = tf.Variable(theta[1], dtype=tf.float32, name="Weight")
theta2 = tf.Variable(theta[0], dtype=tf.float32, name="bias")

#declare placeholders for sample data
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)



#start a new tensorflow session
sess = tf.Session()

#initialize all variables
init = tf.global_variables_initializer()
sess.run(init)

#declare hypothesis
h = theta1*x+theta2

#declare squred error
squared_error = (h-y)**2
#declare aggregrate of all squared errors to be minimized
square_sum = (1.0/(2*float(m)))*tf.reduce_sum(squared_error, 0)

#run the above declared equations(tensors)
print sess.run(square_sum, {x:x_train, y:y_train})

optimizer = tf.train.GradientDescentOptimizer(0.001)

train = optimizer.minimize(square_sum)

for i in range(10000):
    sess.run(train, {x:x_train, y:y_train})


print sess.run(square_sum, {x:x_train, y:y_train})
print sess.run([theta1, theta2])

predict = sess.run(h, {x:x_train}) #predict values using trained weights

#plot the initial training data and the trained hypothesis

plt.plot(x_train, predict)
plt.draw()
plt.pause(10)

















