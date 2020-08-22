from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
import numpy as np

model = Sequential()
model.add(Dense(2,kernel_initializer="truncated_normal",input_dim=2))
model.add(Activation("relu"))
'''
model.add(Dense(2,kernel_initializer="truncated_normal"))
model.add(Activation("relu"))
model.add(Dense(4,kernel_initializer="truncated_normal"))
model.add(Activation("relu"))
'''
model.add(Dense(1,kernel_initializer="truncated_normal"))
model.add(Activation("relu"))


model.summary()
    
model.compile(optimizer=Adam(lr = 0.0005), loss='mse', metrics=['accuracy'])


x = np.array([[0,0],[0,1],[1,0],[1,1]])

y = np.array([1,0,0,1])


model.fit(x,y,batch_size=1,epochs=100000)
