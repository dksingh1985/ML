import tensorflow as tf
import numpy as np
from collections import OrderedDict 
import math

def conv2d(x,filter_size,channels,features):
    print("Conv2d x : " , x.get_shape())
    with tf.name_scope("Conv2d"):
        weight = tf.Variable(tf.truncated_normal([filter_size, filter_size, channels, features], stddev=0.01),name="Weight")
        bias = tf.Variable(tf.constant(0.1, shape=[features]), name="Bias")
        c_0 = tf.nn.conv2d(x,weight,strides=[1,1,1,1],padding="SAME")
        c_1 = tf.nn.bias_add(c_0,bias)
        c_2 = tf.nn.relu(c_1)
        #c_3 = tf.nn.dropout(c_1,keep_prob)
        return c_2,weight,bias

def conv2d_transpose(x,feature):
    print("Conv2d Transposex : " , x.get_shape())
    with tf.name_scope("Conv2d_Transpose"):
        x_shape = x.get_shape().as_list() #tf.shape(x)
        o_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
        weight = tf.Variable(tf.truncated_normal([2,2,feature//2,feature], stddev=0.01),name="Weight")
        bias = tf.Variable(tf.constant(0.1, shape=[feature//2]), name="Bias")
        ct_0 = tf.nn.conv2d_transpose(x,weight,o_shape, strides=[1,2,2,1], padding="SAME")
        ct_1  = tf.nn.bias_add(ct_0,bias)
        ct_2  = tf.nn.relu(ct_1)
        return ct_2,weight,bias

def max_pool(x):
    print("Max Pool x : " , x.get_shape())
    mp_0 = tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1],padding="SAME")
    return mp_0

def crop_and_concat(x1,x2):
    print("Crop ans Concat x1 & x2 : " , x1.get_shape().as_list(), " : " , x2.get_shape().as_list())
    with tf.name_scope("crop_and_concat"):
        x1_shape = x1.get_shape().as_list()
        x2_shape = x2.get_shape().as_list()
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        cc_0 = tf.concat([x2,x1_crop], 3)
        #print("CC_0 : ", cc_0.get_shape())
        return cc_0

def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize

def cross_entropy(y_,output_map):
    return tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")

def create_unet(x,depth,class_count):
    weights = OrderedDict()
    baises = OrderedDict()
    layers = OrderedDict()
    starting_feature = 32
    filter_size = 7

    #Down layers 1
    feature = starting_feature  #32,101
    layers["L1_C1"], weights["L1_C1"], baises["L1_C1"] = conv2d(x,filter_size,1,feature)
    layers["L1_C2"], weights["L1_C2"], baises["L1_C2"] = conv2d(layers["L1_C1"],filter_size,feature,feature)
    layers["L1_P3"] = max_pool(layers["L1_C2"])

    #Down layers 2
    feature = int(2*feature)     #64,51
    layers["L2_C1"], weights["L2_C1"], baises["L2_C1"] = conv2d(layers["L1_P3"],filter_size,int(feature/2),feature)
    layers["L2_C2"], weights["L2_C2"], baises["L2_C2"] = conv2d(layers["L2_C1"],filter_size,feature,feature)
    layers["L2_P3"] = max_pool(layers["L2_C2"])
    
    #Down layers 3
    feature = int(2*feature)     #128,26
    layers["L3_C1"], weights["L3_C1"], baises["L3_C1"] = conv2d(layers["L2_P3"],filter_size,int(feature/2),feature)
    layers["L3_C2"], weights["L3_C2"], baises["L3_C2"] = conv2d(layers["L3_C1"],filter_size,feature,feature)
    layers["L3_P3"] = max_pool(layers["L3_C2"])
    
    #Bottom layers 4
    feature = int(2*feature)     #256,13
    layers["L4_C1"], weights["L4_C1"], baises["L4_C1"] = conv2d(layers["L3_P3"],filter_size,int(feature/2),feature)
    layers["L4_C2"], weights["L4_C2"], baises["L4_C2"] = conv2d(layers["L4_C1"],filter_size,feature,feature)
    
    #Up layers 3
    feature = int(feature/2)     #128,26
    layers["L3_T1"], weights["L3_T1"], baises["L3_T1"] = conv2d_transpose(layers["L4_C2"],int(feature*2))
    layers["L3_J2"] = crop_and_concat(layers["L3_T1"],layers["L3_C2"])
    layers["L3_C3"], weights["L3_C3"], baises["L3_C3"] = conv2d(layers["L3_J2"],filter_size,int(feature*2),feature)
    layers["L3_C4"], weights["L3_C4"], baises["L3_C4"] = conv2d(layers["L3_C3"],filter_size,feature,feature)
    
    #Up layers 2
    feature = int(feature/2)     #64,51
    layers["L2_T1"], weights["L2_T1"], baises["L2_T1"] = conv2d_transpose(layers["L3_C4"],int(feature*2))
    layers["L2_J2"] = crop_and_concat(layers["L2_T1"],layers["L2_C2"])
    layers["L2_C3"], weights["L2_C3"], baises["L2_C3"] = conv2d(layers["L2_J2"],filter_size,int(feature*2),feature)
    layers["L2_C4"], weights["L2_C4"], baises["L2_C4"] = conv2d(layers["L2_C3"],filter_size,feature,feature)

    #Up layers 1
    feature = int(feature/2)     #32,101
    layers["L1_T1"], weights["L1_T1"], baises["L1_T1"] = conv2d_transpose(layers["L2_C4"],int(feature*2))
    layers["L1_J2"] = crop_and_concat(layers["L1_T1"],layers["L1_C2"])
    layers["L1_C3"], weights["L1_C3"], baises["L1_C3"] = conv2d(layers["L1_J2"],filter_size,int(feature*2),feature)
    layers["L1_C4"], weights["L1_C4"], baises["L1_C4"] = conv2d(layers["L1_C3"],filter_size,feature,feature)

    #Out layers 0
    layers["out"], weights["out"], baises["out"] = conv2d(layers["L1_C4"],filter_size,feature,class_count)

    print("Y predicted : ", layers["out"].get_shape())

    return layers,weights,baises

class unet(object):
    def __init__(self,x,depth,class_count):

        tf.reset_default_graph()
                
        self.x = x
        self.depth = depth
        self.class_count = class_count
        self.variables = []

        self.x = tf.placeholder("float", shape=[None, None, None, 1], name="x")
        self.y = tf.placeholder("float", shape=[None, None, None, class_count], name="y")
        
        layers,weights,baises = create_unet(x,depth,class_count)

        self.variables.append(weights)
        self.variables.append(baises)

        self.cost = self.__cost(layers["out"])

        self.gradients_node = tf.gradoents(self.cost,self.variables)

        self.cross_entropy = cross_entropy

        with tf.name_scope("results"):
            self.predicter = pixel_wise_softmax(layers["out"])
            self.correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        def _cost(self,logits):
            with tf.name_scope("cost"):
                flat_logits = tf.reshape(logits, [-1, self.n_class])
                flat_labels = tf.reshape(self.y, [-1, self.n_class])
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,labels=flat_labels))
                return loss

        def predict(self,model_path,x_test):
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                self.restore(sess, model_path)
                y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
                prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy})
            return prediction

        def save(self,sess,model_path):
            saver = tf.train.Saver()
            save_path = saver.save(sess, model_path)
            return save_path

        def restore(self, sess, model_path):
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print("Model restored from file: %s" % model_path)


class Trainer(object):

    def __init__(self, net, batch_size=1, verification_batch_size = 4, norm_grads=False):
        self.net = net
        self.batch_size = batch_size
        self.verification_batch_size = verification_batch_size
        self.norm_grads = norm_grads
        self.optimizer = optimizer

    def _get_optimizer(self, training_iters, global_step):
        learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
        decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
        momentum = self.opt_kwargs.pop("momentum", 0.2)

        self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,global_step=global_step,decay_steps=training_iters,decay_rate=decay_rate,staircase=True)

        optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum).minimize(self.net.cost,global_step=global_step)

        return optimizer

    def _initialize(self, training_iters, output_path, restore, prediction_path):
        global_step = tf.Variable(0, name="global_step")
        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]), name="norm_gradients")
        self.optimizer = self._get_optimizer(training_iters, global_step)
        init = tf.global_variables_initializer()
        return init

        

    

    
            
    

    
    

