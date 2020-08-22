from config import *

def get_feature_map():
    model = Sequential()
    #768
    model.add(Conv2D(8,(3,3),padding = 'same',activation="relu",input_shape =(768,768,3)))
    model.add(Conv2D(8,(3,3),padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    #384
    model.add(Conv2D(16,(3,3),padding="same",activation="relu"))
    model.add(Conv2D(16,(3,3),padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    #192
    model.add(Conv2D(32,(3,3),padding="same",activation="relu"))
    model.add(Conv2D(32,(3,3),padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    #96
    model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
    model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    #48
    model.add(Conv2D(128,(3,3),padding="same",activation="relu"))
    model.add(Conv2D(128,(3,3),padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    '''
    #24
    model.add(Conv2D(32,(3,3),padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    #12
    model.add(Conv2D(32,(3,3),padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    #6
    model.add(Conv2D(32,(3,3),padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    #3
    model.add(Conv2D(32,(3,3),padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    '''
    model.add(Flatten())
    #model.add(Dense(1024,activation="relu"))
    model.add(Dense(1024,activation="relu"))
    model.add(Dense(1024,activation="sigmoid"))

              
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr = 1e-4), metrics=["accuracy"])

    model.summary()
    
    return model


def region_proposal_network():
    print()




def unet():
    inputs = Input(IMAGE_SIZE_UNET,IMAGE_SIZE_UNET,3)
    
    #Down Layer 1, 96
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    dl_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_1)
    dl_pool_1 = MaxPooling2D(pool_size=(2, 2))(dl_conv_1)

    #Down Layer 2, 48
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_pool_1)
    dl_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_2)
    dl_pool_2 = MaxPooling2D(pool_size=(2, 2))(dl_drop_2)

    #Down Layer 3, 24
    dl_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_pool_2)
    dl_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_3)
    dl_pool_3 = MaxPooling2D(pool_size=(2, 2))(dl_drop_3)

    #Down Layer 4, 12
    dl_conv_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_pool_3)
    dl_conv_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_4)
    dl_pool_4 = MaxPooling2D(pool_size=(2, 2))(dl_drop_4)

    #Down Layer 5, 6
    dl_conv_5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_pool_4)
    dl_conv_5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_conv_5)
    dl_pool_5 = MaxPooling2D(pool_size=(2, 2))(dl_drop_5)

    #Bottom Layer 6, 3
    bl_conv_8 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dl_pool_4)
    bl_conv_8 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(bl_conv_8)
    bl_drop_8 = Dropout(DROPOUT)(bl_conv_8)

    #Up Layer 5, 6
    ul_upsamp_5 = UpSampling2D(size = (2,2))(ul_conv_6)
    ul_concat_5 = concatenate([dl_drop_5,ul_upsamp_5], axis = 3)
    ul_conv_5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_concat_5)
    ul_conv_5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_5)

    #Up Layer 4, 12
    ul_upsamp_4 = UpSampling2D(size = (2,2))(bl_conv_8)
    ul_concat_4 = concatenate([dl_drop_4,ul_upsamp_4], axis = 3)
    ul_conv_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_concat_4)
    ul_conv_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_4)
    
    #Up Layer 3, 24
    ul_upsamp_3 = UpSampling2D(size = (2,2))(ul_conv_4)
    ul_concat_3 = concatenate([dl_drop_3,ul_upsamp_3],axis = 3)
    ul_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_concat_3)
    ul_conv_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_3)
    
    #Up Layer 2, 48
    ul_upsamp_2 = UpSampling2D(size = (2,2))(ul_conv_3)
    ul_concat_2 = concatenate([dl_drop_2,ul_upsamp_2],axis = 3)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_concat_2)
    ul_conv_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_2)
    
    #up Layer 1, 96
    ul_upsamp_1 = UpSampling2D(size = (2,2))(ul_conv_2)
    ul_concat_1 = concatenate([dl_drop_1,ul_upsamp_1], axis = 3)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_concat_1)
    ul_conv_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ul_conv_1)
    
    ul_conv_0 = Conv2D(1, 1, activation = 'sigmoid')(ul_conv_1)

    model = Model(inputs=inputs, outputs=ul_conv_0)

    #model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.01,momentum=0.01),metrics=["accuracy"])
    #model.compile(optimizer = Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8), loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer=Adam(lr = 1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    model.summary()

    return model
    
