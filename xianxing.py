import numpy as np
import tensorflow as tf 
data_x = np.linspace(0,10,30)
data_y = data_x *3+7+np.random.normal(0,1,30)

w=tf.Variable(1.,name="weight")
b=tf.Variable(0.,name="bias")

x = tf.placeholder(tf.float32,shape=None)
y = tf.placeholder(tf.float32,shape=None)

pred = tf.multiply(x,w)+b

loss = tf.reduce_sum(tf.squared_difference(pred,y))

learn_rate = 0.0001
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for i in range(10000):
		sess.run(train_step,feed_dict={y:data_y,x:data_x})
	
		if i % 1000 == 0:
			print(sess.run([loss,w,b],feed_dict={y:data_y,x:data_x}))


