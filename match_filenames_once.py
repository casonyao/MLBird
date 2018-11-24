import tensorflow as tf 
 
directory = "D:\MachineLearn\*hello*.py"
file_names = tf.train.match_filenames_once(directory)
 
init = (tf.global_variables_initializer(), tf.local_variables_initializer())
 
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(file_names))
