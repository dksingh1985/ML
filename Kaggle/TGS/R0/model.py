import tensorflow as tf
import unet as un
from PIL import Image
import os
import numpy as np
import random
import pandas as pd
import datetime

learning_rate = 0.1
tareget_cost = 0.1
training_batch_size = 10
model_save_path = "D:\\Data\\Kaggle\\TGS\\Model_R0"


x_ = tf.placeholder(tf.float32,shape=[training_batch_size,101,101,1],name="x-input")
y_ = tf.placeholder(tf.float32,shape=[training_batch_size,101,101,1],name="y-input")
z_ = tf.placeholder(tf.float32,shape=[training_batch_size,910],name="Z-input")
depth_df = pd.read_csv("D:\\Data\\Kaggle\\TGS\\depths.csv")
global_step = tf.Variable(0, name="global_step")

in_image_T = np.zeros([training_batch_size,101,101,1])
out_image_T = np.zeros([training_batch_size,101,101,1])
depth = np.zeros([training_batch_size,910])

imgList =  os.listdir("D:\\Data\\Kaggle\\TGS\\train\\T")
imgList =  random.sample(os.listdir("D:\\Data\\Kaggle\\TGS\\train\\T"),training_batch_size)
random.shuffle(imgList)

print("Reading new set of file(s):",training_batch_size,", Learning Rate:", learning_rate, ", Target Cost:", tareget_cost, ", Time Date:", datetime.datetime.now())
for j in range(training_batch_size):
    #print("Reading file : ", i , " = ", imgList[i])
    in_image_T[j,]  = np.multiply(0.003921568627451,Image.open("D:\\Data\\Kaggle\\TGS\\train\\T\\" + imgList[j]).convert("L")).reshape([101,101,1])
    out_image_T[j,] = np.multiply(0.003921568627451,Image.open("D:\\Data\\Kaggle\\TGS\\train\\R\\" + imgList[j]).convert("L")).reshape([101,101,1])
    depth[j,(depth_df[depth_df.id == imgList[j].split(".")[0]].iloc[0].z) - 50] = 1

l,w,b = un.create_unet(x_,5,1)

#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=l["out"],labels=y_))
loss = tf.losses.mean_squared_error(y_,l["out"],weights=1.0)

learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,global_step=global_step,decay_steps=10,decay_rate=0.95,staircase=True)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_node,momentum=0.01).minimize(loss,global_step=global_step)

#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


init = tf.global_variables_initializer()

saver = tf.train.Saver()

sess = tf.Session()
writer = tf.summary.FileWriter("C:\\TEMP\\tf\\TGS_Model_logs", tf.get_default_graph())
sess.run(init)

if os.path.exists(model_save_path + "\\checkpoint"):
    saver.restore(sess, model_save_path + "\\model.ckpt")

cost_reff = sess.run(loss, feed_dict={x_: in_image_T, y_: out_image_T})
print("# Cost : ", cost_reff)

print('Traning Started')
print('0 #',end='')
for i in range(1000000):
    sess.run(optimizer,feed_dict={x_:in_image_T,y_:out_image_T,z_: depth})
    print( '|',end='')

    if (i % 10 == 0) and (i > 0):
        save_path = saver.save(sess, model_save_path + "\\model.ckpt")
        #print("Model saved in path: %s" % save_path)
        cost_reff = sess.run(loss, feed_dict={x_: in_image_T, y_: out_image_T,z_: depth})
        print("# Cost : ", cost_reff,", Time:", datetime.datetime.now())
        if (i % 100 == 0):
            #training_batch_size += 10
            in_image_T = np.zeros([training_batch_size,101,101,1])
            out_image_T = np.zeros([training_batch_size,101,101,1])
            depth = np.zeros([training_batch_size,910])
            imgList =  os.listdir("D:\\Data\\Kaggle\\TGS\\train\\T")
            imgList =  random.sample(os.listdir("D:\\Data\\Kaggle\\TGS\\train\\T"),training_batch_size)
            random.shuffle(imgList)
            print("Reading new set of file(s):",training_batch_size,", Learning Rate:", learning_rate, ", Target Cost:", tareget_cost, ", Time Date:", datetime.datetime.now())
            for j in range(training_batch_size):
                #print("Reading file : ", i , " = ", imgList[i])
                in_image_T[j,] =  np.multiply(1.0,Image.open("D:\\Data\\Kaggle\\TGS\\train\\T\\" + imgList[j]).convert("L")).reshape([101,101,1])
                out_image_T[j,] = np.multiply(1.0,Image.open("D:\\Data\\Kaggle\\TGS\\train\\R\\" + imgList[j]).convert("L")).reshape([101,101,1])
                depth[j,(depth_df[depth_df.id == imgList[j].split(".")[0]].iloc[0].z) - 50] = 1
        print(i,'#',end='')
        
        




