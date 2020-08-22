import tensorflow as tf
from tensorflow.keras import Model,Input
from tensorflow.keras.layers import Dense,Conv2D,ConvLSTM2D,MaxPooling2D,TimeDistributed,Flatten
import numpy as np

DROPOUT = 0.3
H_UNIT = 4

def get_model(image_size):
    
    encoder_inputs = Input(shape=(1024,image_size,image_size,1))
    #512
    encoder_convL_1=TimeDistributed(Conv2D(H_UNIT, 3, activation = 'sigmoid', padding = 'same'))
    convL = encoder_convL_1(encoder_inputs)
    #encoder_convL_2=TimeDistributed(Conv2D(H_UNIT, 3, activation = 'sigmoid', padding = 'same'))
    #convL = encoder_convL_2(convL)
    convL = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(convL)
    
    #256
    encoder_convL_3=TimeDistributed(Conv2D(H_UNIT*2, 3, activation = 'sigmoid', padding = 'same'))
    convL = encoder_convL_3(convL)
    #encoder_convL_4=TimeDistributed(Conv2D(H_UNIT*2, 3, activation = 'sigmoid', padding = 'same'))
    #convL = encoder_convL_4(convL)
    convL = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(convL)
    
    #128
    encoder_convL_5=TimeDistributed(Conv2D(H_UNIT*4, 3, activation = 'sigmoid', padding = 'same'))
    convL = encoder_convL_5(convL)
    #encoder_convL_6=TimeDistributed(Conv2D(H_UNIT*4, 3, activation = 'sigmoid', padding = 'same'))
    #convL = encoder_convL_6(convL)
    convL = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(convL)
    
    #64
    #encoder_convL_7=TimeDistributed(Conv2D(H_UNIT*8, 3, activation = 'sigmoid', padding = 'same'))
    #convL = encoder_convL_7(convL)
    #encoder_convL_8=TimeDistributed(Conv2D(H_UNIT*8, 3, activation = 'sigmoid', padding = 'same'))
    #convL = encoder_convL_8(convL)
    #convL = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(convL)
    
    #32
    encoder_convL_9=ConvLSTM2D(H_UNIT*8, 3, activation = 'sigmoid', padding = 'same', return_sequences=True)
    convL = encoder_convL_9(convL)
    encoder_convL_10=ConvLSTM2D(H_UNIT*8, 3, activation = 'sigmoid', padding = 'same', return_sequences=True)
    convL = encoder_convL_10(convL)
    convL = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(convL)
    
    #16
    #encoder_convL_11=ConvLSTM2D(H_UNIT*8, 1, activation = 'sigmoid', padding = 'same', return_sequences=True)
    #convL = encoder_convL_11(convL)
    #encoder_convL_12=ConvLSTM2D(H_UNIT*4, 1, activation = 'sigmoid', padding = 'same', return_sequences=True)
    #convL = encoder_convL_12(convL)
    #encoder_convL_13=ConvLSTM2D(H_UNIT*2, 1, activation = 'sigmoid', padding = 'same', return_sequences=True)
    #convL = encoder_convL_13(convL)
    #encoder_convL_14=ConvLSTM2D(H_UNIT, 1, activation = 'sigmoid', padding = 'same', return_sequences=True)
    #convL = encoder_convL_14(convL)
    encoder_convL_15=ConvLSTM2D(1, 1, activation = 'sigmoid', padding = 'same', return_sequences=True)
    convL = encoder_convL_15(convL)
    
    denL = Flatten()(convL)
    outL = Dense(140, activation="relu")(denL)
    
    model = Model(inputs=encoder_inputs, outputs=outL)
    
    #model.compile(optimizer = Adam(lr = 1e-6, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    return model

m = get_model(256)
m.save('D:\\Data\\OSIC\\model.hd5')

#m.predict(np.ones((1,1024,512,512,1)))
    

