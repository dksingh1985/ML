import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

x_data = pd.read_csv('D:\\data\\Kaggle\\digit-recognizer\\test.csv')
x_data = x_data / 255
x_data[x_data < 0.5] = 0
x_data[x_data > 0 ] = 1
x_data = np.resize(x_data,(x_data.shape[0],28,28,1))

print("===>",x_data.shape)


model = Sequential()
model.add(Conv2D(8,kernel_size=3,padding="same",input_shape=(28,28,1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(16,kernel_size=3,padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
#model.add(Dropout(0.1))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.load_weights("D:\\data\\Kaggle\\digit-recognizer\\model.h5")


predict = model.predict(x_data)
y_data = np.argmax(predict,axis=1)
y_data = pd.DataFrame(y_data)
y_data.to_csv('D:\\data\\Kaggle\\digit-recognizer\\test_out.csv')
#saveResult2(PRED_MASK_PATH,imgs,predict,i,BATCH_SIZE)
    
