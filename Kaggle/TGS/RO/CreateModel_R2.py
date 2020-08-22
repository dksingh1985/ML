import tensorflow as tf
from PIL import Image
import os
import numpy as np
import random
import pandas as pd
import datetime

filter_size = 5
pool_size = 2
l1_filter_count = 4
l2_filter_count = 16
training_batch_size = 32
tareget_cost = 0.1
learning_rate = 0.01
model_save_path = "D:\\Data\\Kaggle\\TGS\\Model_R2"
depth_df = pd.read_csv("D:\\Data\\Kaggle\\TGS\\depths.csv")


x_ = tf.placeholder(tf.float32,shape=[None,101,101,1],name="x-input")
y_ = tf.placeholder(tf.float32,shape=[None,10201],name="y-input")
z_ = tf.placeholder(tf.float32,shape=[None,910],name="Z-input")

l1_c_f = tf.Variable(tf.truncated_normal([filter_size,filter_size,1,l1_filter_count],stddev=0.05),name="Filter-Conv1")
l2_c_f = tf.Variable(tf.truncated_normal([filter_size,filter_size,l1_filter_count,l2_filter_count],stddev=0.05),name="Filter-Conv2")

l2_f_w = tf.Variable(tf.truncated_normal([910,10201],0,1),name="Weight-2")
l3_f_w = tf.Variable(tf.truncated_normal([26*26*l2_filter_count,10201],0,1),name="Weight-3")
l4_f_w = tf.Variable(tf.truncated_normal([10201,10201],0,1),name="Weight-4")

l1_c_b = tf.Variable(tf.truncated_normal([l1_filter_count],0,1),name="Bias-Conv1")
l2_c_b = tf.Variable(tf.truncated_normal([l2_filter_count],0,1),name="Bias-Conv2")
l3_f_b = tf.Variable(tf.truncated_normal([10201],0,1),name="Bias-3")
l4_f_b = tf.Variable(tf.truncated_normal([10201],0,1),name="Bias-4")

#Layer 1a
Layer1_out_f = tf.nn.conv2d(input=x_,filter=l1_c_f,strides=[1,1,1,1],padding="SAME") + l1_c_b
Layer1_out_p = tf.nn.max_pool(value=Layer1_out_f,ksize=[1,pool_size,pool_size,1],strides=[1,pool_size,pool_size,1],padding="SAME")
Layer1_out_a = tf.nn.relu(Layer1_out_p)

#Layer 2
Layer2_out_f = tf.nn.conv2d(input=Layer1_out_a,filter=l2_c_f,strides=[1,1,1,1],padding="SAME") + l2_c_b
Layer2_out_p = tf.nn.max_pool(value=Layer2_out_f,ksize=[1,pool_size,pool_size,1],strides=[1,pool_size,pool_size,1],padding="SAME")
Layer2_out_a = tf.nn.relu(Layer2_out_p)

layer_flat = tf.reshape(Layer2_out_a, [-1, 26*26*l2_filter_count])

#Layer 3
Layer3_out = tf.nn.relu((tf.matmul(layer_flat,l3_f_w) + (tf.matmul(z_,l2_f_w))) + l3_f_b)

#Layer 4
Layer4_out = tf.nn.relu(tf.matmul(Layer3_out,l4_f_w) + l4_f_b)

#cost = tf.reduce_mean((( y_ * tf.log(s3)) + ((1 - y_) * tf.log(1.0 - s3))) * -1)
cost = tf.reduce_mean(tf.square(tf.subtract(y_,Layer4_out)))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

imgList =  os.listdir("D:\\Data\\Kaggle\\TGS\\train\\T")
#imgList =  random.sample(os.listdir("D:\\Data\\Kaggle\\TGS\\train\\T"),training_batch_size)
#random.shuffle(imgList)
training_batch_size = len(imgList)

in_image_T = np.zeros([len(imgList),101,101,1])
out_image_T = np.zeros([len(imgList),10201])
depth = np.zeros([len(imgList),910])

print("Reading new set of file(s):",training_batch_size,", Learning Rate:", learning_rate, ", Target Cost:", tareget_cost, ", Time Date:", datetime.datetime.now())
for i in range(len(imgList)):
    #print("Reading file : ", i , " = ", imgList[i])
    in_image_T[i,] = np.multiply(0.003921568627451,Image.open("D:\\Data\\Kaggle\\TGS\\train\\T\\" + imgList[i]).convert("L")).reshape([101,101,1])
    out_image_T[i,] = np.multiply(0.003921568627451,Image.open("D:\\Data\\Kaggle\\TGS\\train\\R\\" + imgList[i]).convert("L")).reshape([-1])
    depth[i,(depth_df[depth_df.id == imgList[i].split(".")[0]].iloc[0].z) - 50] = 1
    
init = tf.global_variables_initializer()

saver = tf.train.Saver()

sess = tf.Session()
writer = tf.summary.FileWriter("C:\\TEMP\\tf\\TGS_Model_logs", tf.get_default_graph())
sess.run(init)

if os.path.exists(model_save_path + "\\checkpoint"):
    saver.restore(sess, model_save_path + "\\model.ckpt")

cost_reff = sess.run(cost, feed_dict={x_: in_image_T, y_: out_image_T,z_: depth})
print("# Cost : ", cost_reff)

print('Traning Started')
print('000 #',end='')
for i in range(1000000):
    sess.run(train_step,feed_dict={x_:in_image_T,y_:out_image_T,z_: depth})
    print( '|',end='')

    if (i % 10 == 0) and (i > 0):
        save_path = saver.save(sess, model_save_path + "\\model.ckpt")
        #print("Model saved in path: %s" % save_path)
        cost_reff = sess.run(cost, feed_dict={x_: in_image_T, y_: out_image_T,z_: depth})
        print("# Cost : ", cost_reff,", Time:", datetime.datetime.now())
        print(i,'#',end='')
        
        #print('Epoch ', i)
        #print('s3 ', sess.run(s3, feed_dict={x_: in_image_T, y_: out_image_T}))
        #print('w1 ', sess.run(w1))
        #print('b1 ', sess.run(b1))
        #print('w2 ', sess.run(w2))
        #print('b2 ', sess.run(b2))
        #print('w3 ', sess.run(w3))
        #print('b3 ', sess.run(b3))
        #print('Epoch ', i, 'cost ', sess.run(cost, feed_dict={x_: in_image_T, y_: out_image_T}))

#print('s3 ', sess.run(s3, feed_dict={x_: XOR_X, y_: XOR_Y}))
        
