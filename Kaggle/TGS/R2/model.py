import numpy as np 
import os
import numpy as np
from keras.models import *
from keras.layers import Conv2D,MaxPooling2D,Dropout,merge,UpSampling2D,concatenate,Activation,Dense,Flatten
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as kb

IMAGE_SIZE = 96
DROPOUT = 0

def unet(pretrained_weights = None,input_size = (96,96,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(DROPOUT)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(DROPOUT)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7],axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8],axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
	
def unet2(pretrained_weights = None,input_size = (96,96,3)):
    inputs = Input(input_size)
    
    #Down Layer 1, 96
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
    dl_drop_1 = Dropout(DROPOUT)(dl_conv_1)
    dl_pool_1 = MaxPooling2D(pool_size=(2, 2))(dl_drop_1)

    print("dl_drop_1: ", kb.int_shape(dl_drop_1)) 
    
    #Down Layer 2, 48
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_pool_1)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_drop_2 = Dropout(DROPOUT)(dl_conv_2)
    dl_pool_2 = MaxPooling2D(pool_size=(2, 2))(dl_drop_2)

    print("dl_drop_2: ", kb.int_shape(dl_drop_2))
    
    #Down Layer 3, 24
    dl_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_pool_2)
    dl_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_3)
    dl_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_3)
    dl_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_3)
    dl_drop_3 = Dropout(DROPOUT)(dl_conv_3)
    dl_pool_3 = MaxPooling2D(pool_size=(2, 2))(dl_drop_3)

    print("dl_drop_3: ", kb.int_shape(dl_drop_3))

    #Down Layer 4, 12
    dl_conv_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_pool_3)
    dl_conv_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_4)
    dl_drop_4 = Dropout(DROPOUT)(dl_conv_4)
    dl_pool_4 = MaxPooling2D(pool_size=(2, 2))(dl_drop_4)

    print("dl_drop_4: ", kb.int_shape(dl_drop_4))
    
    #Down Layer 5, 6
    dl_conv_5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_pool_4)
    #conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    dl_conv_5 = Dropout(DROPOUT)(dl_conv_5)
    dl_drop_5 = Dropout(DROPOUT)(dl_conv_5)
    dl_pool_5 = MaxPooling2D(pool_size=(2, 2))(dl_drop_5)

    print("dl_drop_5: ", kb.int_shape(dl_drop_5))
    
    #Bottom Layer 6, 3
    bl_conv_6 = Conv2D(2048, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_pool_5)
    bl_drop_6 = Dropout(DROPOUT)(bl_conv_6)

    print("bl_drop_6: ", kb.int_shape(bl_drop_6))

    #Up Layer 5, 6
    ul_upsamp_5 = UpSampling2D(size = (2,2))(bl_drop_6)
    ul_upsamp_5 = Conv2D(1024, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_upsamp_5)
    print("ul_upsamp_5:",kb.int_shape(ul_upsamp_5))
    ul_concat_5 = concatenate([dl_drop_5,ul_upsamp_5], axis = 3)
    ul_conv_5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_concat_5)

    #Up Layer 4, 12
    ul_upsamp_4 = UpSampling2D(size = (2,2))(ul_conv_5)
    ul_upsamp_4 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_upsamp_4)
    print("ul_upsamp_4:",kb.int_shape(ul_upsamp_4))
    ul_concat_4 = concatenate([dl_drop_4,ul_upsamp_4], axis = 3)
    ul_conv_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_concat_4)
    ul_conv_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_4)
    
    #Up Layer 3, 25
    ul_upsamp_3 = UpSampling2D(size = (2,2))(ul_conv_4)
    ul_upsamp_3 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_upsamp_3)
    '''
    print("ul_upsamp_3:",kb.int_shape(ul_upsamp_3))
    temp_ul_upsamp_3 = ul_upsamp_3[:,:,23:24,:]
    print("temp_ul_upsamp_3:",kb.int_shape(temp_ul_upsamp_3))
    ul_upsamp_3 = concatenate([ul_upsamp_3,temp_ul_upsamp_3],axis = 2)
    print("ul_upsamp_3:",kb.int_shape(ul_upsamp_3))
    temp_ul_upsamp_3 = ul_upsamp_3[:,23:24,:,:]
    print("temp_ul_upsamp_3:",kb.int_shape(temp_ul_upsamp_3))
    ul_upsamp_3 = concatenate([ul_upsamp_3,temp_ul_upsamp_3],axis = 1)
    print("ul_upsamp_3:",kb.int_shape(ul_upsamp_3))
    '''
    ul_concat_3 = concatenate([dl_drop_3,ul_upsamp_3],axis = 3)
    ul_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_concat_3)
    ul_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_3)
    ul_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_3)
    ul_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_3)
    
    #Up Layer 2, 50
    ul_upsamp_2 = UpSampling2D(size = (2,2))(ul_conv_3)
    ul_upsamp_2 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_upsamp_2)
    print("ul_upsamp_2:",kb.int_shape(ul_upsamp_2))
    ul_concat_2 = concatenate([dl_drop_2,ul_upsamp_2],axis = 3)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_concat_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    

    #up Layer 1, 101
    ul_upsamp_1 = UpSampling2D(size = (2,2))(ul_conv_2)
    ul_upsamp_1 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_upsamp_1)
    '''
    print("ul_upsamp_1:",kb.int_shape(ul_upsamp_1))
    temp_ul_upsamp_1 = ul_upsamp_1[:,:,99:100,:]
    print("temp_ul_upsamp_1:",kb.int_shape(temp_ul_upsamp_1))
    ul_upsamp_1 = concatenate([ul_upsamp_1,temp_ul_upsamp_1],axis = 2)
    print("ul_upsamp_1:",kb.int_shape(ul_upsamp_1))
    temp_ul_upsamp_1 = ul_upsamp_1[:,99:100,:,:]
    print("temp_ul_upsamp_1:",kb.int_shape(temp_ul_upsamp_1))
    ul_upsamp_1 = concatenate([ul_upsamp_1,temp_ul_upsamp_1],axis = 1)
    print("ul_upsamp_1:",kb.int_shape(ul_upsamp_1))
    '''
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
    
    ul_conv_0 = Conv2D(1, 1, activation = 'sigmoid')(ul_conv_1)

    model = Model(inputs=inputs, outputs=ul_conv_0)

    #model.compile(optimizer=Adam(lr = 1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    model.summary()

    return model
	
def net2(input_size = (101,101,1)):
    #Dropout=0.1
    #101
    model = Sequential()
    model.add(Conv2D(32, (11,11), padding='same',activation='relu', input_shape=input_size))
    model.add(Conv2D(32, (11,11),padding='same',activation='relu'))
    model.add(Conv2D(32, (11,11),padding='same',activation='relu'))
    model.add(Conv2D(32, (11,11),padding='same',activation='relu'))
    model.add(Conv2D(32, (11,11),padding='same',activation='relu'))
    model.add(Conv2D(32, (11,11),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(Dropout(DROPOUT))

    #51
    model.add(Conv2D(64, (1,1),  padding='same',activation='relu'))
    model.add(Conv2D(64, (9,9), padding='same',activation='relu'))
    model.add(Conv2D(64, (9,9), padding='same',activation='relu'))
    model.add(Conv2D(64, (9,9), padding='same',activation='relu'))
    model.add(Conv2D(64, (9,9), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(Dropout(DROPOUT))

    #26
    model.add(Conv2D(128, (1,1), padding='same',activation='relu'))
    model.add(Conv2D(128, (7,7), padding='same',activation='relu'))
    model.add(Conv2D(128, (7,7), padding='same',activation='relu'))
    model.add(Conv2D(128, (7,7), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(Dropout(DROPOUT))

    #13
    model.add(Conv2D(256, (1,1), padding='same',activation='relu'))
    model.add(Conv2D(256, (5,5), padding='same',activation='relu'))
    model.add(Conv2D(256, (5,5), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(Dropout(DROPOUT))

    #7
    model.add(Conv2D(512, (1,1), padding='same',activation='relu'))
    model.add(Conv2D(512, (3,3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(Dropout(DROPOUT))

    
    #4
    model.add(Conv2D(1024, (1,1), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(Dropout(DROPOUT))
    
    
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(DROPOUT))

    model.add(Dense(2,activation='softmax'))

    model.compile(loss="mae", optimizer=Adam(lr = 1e-4), metrics=["accuracy"])

    model.summary()
    
    return model

def net3(input_size = (96,96,1)):

    inputs = Input(input_size)
    
    #Layer-1 96,1
    conv11 = Conv2D(32, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv12 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv13 = Conv2D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv14 = Conv2D(32, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv15 = Conv2D(32, 9, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    ccat13 = concatenate([conv11,conv12,conv13,conv14,conv15],axis=3)
    mpol14 = MaxPooling2D(pool_size=(2, 2))(ccat13)
    drop15 = Dropout(DROPOUT)(mpol14)

    #Layer-2 48, 2X32
    conv21 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop15)
    conv22 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop15)
    conv23 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop15)
    conv24 = Conv2D(64, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop15)
    conv25 = Conv2D(64, 9, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop15)
    ccat23 = concatenate([conv21,conv22,conv23,conv24,conv25],axis=3)
    mpol24 = MaxPooling2D(pool_size=(2, 2))(ccat23)
    drop25 = Dropout(DROPOUT)(mpol24)

    #Layer-3 24, 2X64
    conv31 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop25)
    conv32 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop25)
    conv33 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop25)
    conv34 = Conv2D(128, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop25)
    conv35 = Conv2D(128, 9, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop25)
    ccat33 = concatenate([conv31,conv32,conv33,conv34,conv35],axis=3)
    mpol34 = MaxPooling2D(pool_size=(2, 2))(ccat33)
    drop35 = Dropout(DROPOUT)(mpol34)

    #Layer-4 12, 2X128
    conv41 = Conv2D(256, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop35)
    conv42 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop35)
    conv43 = Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop35)
    conv44 = Conv2D(256, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop35)
    conv45 = Conv2D(256, 9, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop35)
    ccat43 = concatenate([conv41,conv42,conv43,conv44,conv45],axis=3)
    mpol44 = MaxPooling2D(pool_size=(2, 2))(ccat43)
    drop45 = Dropout(DROPOUT)(mpol44)

    #Layer-5 6, 2X256
    conv51 = Conv2D(512, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop45)
    conv52 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop45)
    conv53 = Conv2D(512, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop45)
    conv54 = Conv2D(512, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop45)
    conv55 = Conv2D(512, 9, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop45)
    ccat53 = concatenate([conv51,conv52,conv53,conv54,conv55],axis=3)
    mpol54 = MaxPooling2D(pool_size=(2, 2))(ccat53)
    drop55 = Dropout(DROPOUT)(mpol54)

    #Layer-6 3, 2X512 = 
    flat61 = Flatten()(mpol54)
    dens62 = Dense(3072,activation='relu')(flat61)
    dens63 = Dense(2,activation='softmax')(dens62)
    
    model = Model(input = inputs, output = dens63)
    
    model.compile(loss="mae", optimizer=SGD(lr=0.01,momentum=0.01),metrics=["accuracy"])

    model.summary()
    
    return model
    





