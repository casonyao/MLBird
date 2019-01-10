# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:55:23 2019

@author: Administrator
"""
import tensorflow as tf 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt 
#获取mnist的数据集合
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist)
plt.imshow(mnist.train.images[69].reshape(28,28))


print(mnist.train.labels[0])

x = tf.placeholder("float",[None,784])
y_ = tf.placeholder("float",[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#y =tf.nn.softmax(tf.matmul(x,W)+b)
W_DNN= tf.Variable(tf.zeros([120,10]))
input_layer = tf.layers.dense(inputs=x, units=200, activation=tf.nn.relu)
hid_layer_user_1 = tf.layers.dense(inputs=input_layer,units=200,activation=tf.nn.relu)
hid_layer_user_2 = tf.layers.dense(inputs=hid_layer_user_1,units=120,activation=tf.nn.relu)
y = tf.nn.softmax(tf.matmul(hid_layer_user_2,W_DNN)+b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

init =tf.initialize_all_variables()

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i %100 ==0:
            print(sess.run(cross_entropy,feed_dict={x:batch_xs,y_:batch_ys}))
        
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        
