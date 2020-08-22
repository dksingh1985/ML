import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical

x_data = pd.read_csv('D:\\data\\Kaggle\\digit-recognizer\\train.csv')
y_data = x_data.loc[:,'label':'label']
y_data = to_categorical(y_data,10)
x_data = x_data.loc[:,'pixel0':]
x_data = x_data / 255
x_data[x_data < 0.5] = 0
x_data[x_data > 0 ] = 1
#x_data = np.resize(x_data,(-1,28,28))


model = Sequential()
model.add(Dense(784,input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

history = model.fit(x_data,y_data,batch_size=128,epochs=200,verbose=1,validation_split=0.1)

