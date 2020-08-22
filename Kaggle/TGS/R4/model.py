import numpy as np 
import os
import numpy as np
from keras.models import *
from keras.layers import Conv2D,MaxPooling2D,Dropout,merge,UpSampling2D,concatenate,Activation,Dense,Flatten
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as kb

IMAGE_SIZE = 112
INPUT_CHANNEL = 1
DROPOUT = 0.05

def unet2(pretrained_weights = None,input_size = (IMAGE_SIZE,IMAGE_SIZE,INPUT_CHANNEL)):
    inputs = Input(input_size)
    
    #Down Layer 1, 112
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_drop_1 = Dropout(DROPOUT)(dl_conv_1)
    dl_pool_1 = MaxPooling2D(pool_size=(2, 2))(dl_drop_1)

    #Down Layer 2, 56
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_pool_1)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_drop_2 = Dropout(DROPOUT)(dl_conv_2)
    dl_pool_2 = MaxPooling2D(pool_size=(2, 2))(dl_drop_2)

    #Down Layer 3, 28
    dl_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_pool_2)
    dl_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_3)
    dl_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_3)
    dl_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_3)
    dl_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_3)
    dl_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_3)
    dl_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_3)
    dl_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_3)
    dl_drop_3 = Dropout(DROPOUT)(dl_conv_3)
    dl_pool_3 = MaxPooling2D(pool_size=(2, 2))(dl_drop_3)

    #Down Layer 4, 14
    dl_conv_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_pool_3)
    dl_conv_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_4)
    dl_conv_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_4)
    dl_conv_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_4)
    dl_drop_4 = Dropout(DROPOUT)(dl_conv_4)
    dl_pool_4 = MaxPooling2D(pool_size=(2, 2))(dl_drop_4)
    '''
    #Down Layer 5, 8
    dl_conv_5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_pool_4)
    dl_conv_5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_5)
    dl_drop_5 = Dropout(DROPOUT)(dl_conv_5)
    dl_pool_5 = MaxPooling2D(pool_size=(2, 2))(dl_drop_5)

    #Down Layer 6, 4
    dl_conv_6 = Conv2D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_pool_5)
    dl_conv_6 = Conv2D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_6)
    dl_drop_6 = Dropout(DROPOUT)(dl_conv_6)
    dl_pool_6 = MaxPooling2D(pool_size=(2, 2))(dl_drop_6)
    
    #Down Layer 7, 2
    dl_conv_7 = Conv2D(4096, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_pool_6)
    dl_conv_7 = Conv2D(4096, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_7)
    dl_drop_7 = Dropout(DROPOUT)(dl_conv_7)
    dl_pool_7 = MaxPooling2D(pool_size=(2, 2))(dl_drop_7)
    '''
    #Bottom Layer 8, 7
    bl_conv_8 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_pool_4)
    bl_conv_8 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(bl_conv_8)
    bl_drop_8 = Dropout(DROPOUT)(bl_conv_8)
    '''
    #Up Layer 7, 2
    ul_upsamp_7 = UpSampling2D(size = (2,2))(bl_drop_8)
    ul_upsamp_7 = Conv2D(4096, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_upsamp_7)
    ul_concat_7 = concatenate([dl_drop_7,ul_upsamp_7], axis = 3)
    ul_conv_7 = Conv2D(4096, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_concat_7)
    ul_conv_7 = Conv2D(4096, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_7)
    
    #Up Layer 6, 4
    ul_upsamp_6 = UpSampling2D(size = (2,2))(bl_drop_8)
    ul_upsamp_6 = Conv2D(2048, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_upsamp_6)
    ul_concat_6 = concatenate([dl_drop_6,ul_upsamp_6], axis = 3)
    ul_conv_6 = Conv2D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_concat_6)
    ul_conv_6 = Conv2D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_6)

    #Up Layer 5, 8
    ul_upsamp_5 = UpSampling2D(size = (2,2))(ul_conv_6)
    ul_upsamp_5 = Conv2D(1024, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_upsamp_5)
    ul_concat_5 = concatenate([dl_drop_5,ul_upsamp_5], axis = 3)
    ul_conv_5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_concat_5)
    ul_conv_5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_5)
    '''
    #Up Layer 4, 16
    ul_upsamp_4 = UpSampling2D(size = (2,2))(bl_conv_8)
    ul_upsamp_4 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_upsamp_4)
    ul_concat_4 = concatenate([dl_drop_4,ul_upsamp_4], axis = 3)
    ul_conv_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_concat_4)
    ul_conv_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_4)
    ul_conv_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_4)
    ul_conv_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_4)
    
    #Up Layer 3, 32
    ul_upsamp_3 = UpSampling2D(size = (2,2))(ul_conv_4)
    ul_upsamp_3 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_upsamp_3)
    ul_concat_3 = concatenate([dl_drop_3,ul_upsamp_3],axis = 3)
    ul_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_concat_3)
    ul_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_3)
    ul_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_3)
    ul_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_3)
    ul_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_3)
    ul_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_3)
    ul_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_3)
    ul_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_3)
    
    #Up Layer 2, 64
    ul_upsamp_2 = UpSampling2D(size = (2,2))(ul_conv_3)
    ul_upsamp_2 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_upsamp_2)
    ul_concat_2 = concatenate([dl_drop_2,ul_upsamp_2],axis = 3)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_concat_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    
    #up Layer 1, 128
    ul_upsamp_1 = UpSampling2D(size = (2,2))(ul_conv_2)
    ul_upsamp_1 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_upsamp_1)
    ul_concat_1 = concatenate([dl_drop_1,ul_upsamp_1], axis = 3)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_concat_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    
    ul_conv_0 = Conv2D(1, 1, activation = 'sigmoid')(ul_conv_1)

    model = Model(inputs=inputs, outputs=ul_conv_0)

    #model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.01,momentum=0.01),metrics=["accuracy"])
    #model.compile(optimizer = Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8), loss = 'binary_crossentropy', metrics = ['accuracy'])
    #model.compile(optimizer=Adam(lr = 1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    model.summary()

    return model
	
