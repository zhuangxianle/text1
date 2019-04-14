import tensorflow as tf
import os
import numpy as np

x = tf.placeholder("float",[None,784])

w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,w) + b)

y_ = tf.placeholder("float",[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradiantDesentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs,batch_ys = mnist.train.next_batch(100)
  sess.run(train_step,feed_dict={x : batch_xs, y : batch_ys})
