# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:55:23 2019

@author: Administrator
"""
import tensorflow as tf 
import numpy as np 

x_data = np.float32(np.random.rand(2,100))
y_data = np.dot([0.1,0.2],x_data) +0.3  

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1,2],-1,1))
y = tf.matmul(W,x_data) + b 

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(200):
        sess.run(train)
        if(i%20==0):
            print(i ,sess.run(W),sess.run(b))
