import tensorflow as tf
import numpy as np
import data_provider 
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import csv
#要保存后csv格式的文件名

def get_label(filepath):
    label = []
    fr = open(filepath)
    for row in fr.readlines():
        array = row.strip().split(' ')
        if row.strip() == '' or len(array) != 2:
            continue
        vec = []
        for i in array:
            vec.append(int(i))
        label.append(vec)
    fr.close()
    return np.asarray(label, dtype=np.int32)

howtoTraindata = "./howotData/trainData3.txt"
howtoLabelData = "./howotData/trainLabel3.txt" 
howtovalidData ="./howotData/howtovalidCut.txt" 
totalvocbData = "./howotData/totalData.txt"
outPreData = "./howotData/preData.txt"
sentenceDictT=[]
sentenceDictTrain=[]
sentenceDictValid=[]

stopWordsList=[]
with open(totalvocbData,"r",encoding='UTF-8') as R1:
    for line in R1:
        sentenceDictT.append(line)

with open(howtoTraindata,"r",encoding='UTF-8') as R1:
    for line in R1:
        sentenceDictTrain.append(line)

with open(howtovalidData,"r",encoding='UTF-8') as R1:
    for line in R1:
        sentenceDictValid.append(line) 
        
counter = CountVectorizer()
counts = counter.fit_transform(sentenceDictT)

transF=counter.transform(sentenceDictTrain)
validF=counter.transform(sentenceDictValid)

counter.transform(sentenceDictTrain)

train_query=transF.toarray()
valid_query=validF.toarray()
'''
print('countvectorizer词表:\n',counter.vocabulary_)
print('词向量矩阵:\n',train_query)
print('词向量矩阵:\n',valid_query)

print("countsShape",counts.shape)
print("transFShape",transF.shape)
print("validFShape",validF.shape)
'''
train_label = get_label(howtoLabelData)
print(train_label.shape)

def get_train_batch_data(step, BS):
    global train_query, train_label
    start = step * BS
    end = (step + 1) * BS
    return train_query[start:end, :], train_label[start:end,:]


learning_rate = 0.01
training_epochs = 250
batch_size = 100
display_step = 1
X = tf.placeholder(tf.float32,[None,19155]) # mnist图片尺寸为28*28=784
Y = tf.placeholder(tf.float32,[None,2]) # 0-9共9个数字，10分类问题
# 模型权重
W = tf.Variable(tf.zeros([19155,2]))
b = tf.Variable(tf.zeros([2]))
# 构建模型
pred = tf.nn.softmax(tf.matmul(X,W)+b)
# crossentroy
loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred), reduction_indices=1))
# SGD
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# 初始化
init = tf.global_variables_initializer()
myact=tf.argmax(pred,1)


with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver(max_to_keep=4)

    for epoch in range(training_epochs):
        avg_loss = 0.
        total_batch = int(train_query.shape[0]/batch_size)
        #print("total_batch",total_batch)
        for i in range(total_batch-10):
            batch_xs, batch_ys =get_train_batch_data(i,batch_size)
            valid_xs, valid_ys =get_train_batch_data(i+1,batch_size)
            _,l = sess.run([optimizer,loss],feed_dict={X:batch_xs,Y:batch_ys})
            avg_loss += l / total_batch
            
        if (epoch+1)%display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss))
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #print("Accuracy:",sess.run([accuracy],feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
            print("Accuracy:", accuracy.eval({X: valid_xs, Y: valid_ys}))
            #validRes = sess.run(pred,feed_dict={X:valid_query})
            #print("in pred",sess.run(pred,feed_dict={X:valid_query}))
            '''
            res,act = sess.run([pred,myact],feed_dict={X:valid_query})
            print("res",res)
            #act = tf.argmax(, 1)
            print("myact",act)
            with open(outPreData,"w",encoding='UTF-8') as W1:
                W1.write(str(act))
            '''
            #print("pred.shape", act.eval({X:valid_query}))
            #labelRes = tf.argmax(validRes, 1)
    print("Optimization Finished!")
    res,act = sess.run([pred,myact],feed_dict={X:valid_query})
    print("myact",act)
with open(outPreData,"w",encoding='UTF-8') as W1:
    W1.write(str(act[1:-1]))
