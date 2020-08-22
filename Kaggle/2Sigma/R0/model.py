from data import *


def get_model():
    model = Sequential()
    model.add(Dense(100, kernel_initializer="truncated_normal", input_dim=INPUT_PARAMETER))
    model.add(Activation("sigmoid"))
    
    #model.add(Dropout(0.01))

    model.add(Dense(1024, kernel_initializer="truncated_normal"))
    model.add(Activation("sigmoid"))
    model.add(Dropout(0.01))
    model.add(Dense(512, kernel_initializer="truncated_normal"))
    model.add(Activation("sigmoid"))
    model.add(Dropout(0.01))
    model.add(Dense(256, kernel_initializer="truncated_normal"))
    model.add(Activation("sigmoid"))
    model.add(Dropout(0.01))
    model.add(Dense(128, kernel_initializer="truncated_normal"))
    model.add(Activation("sigmoid"))
    model.add(Dropout(0.01))
    model.add(Dense(64, kernel_initializer="truncated_normal"))
    model.add(Activation("sigmoid"))
    model.add(Dense(32, kernel_initializer="truncated_normal"))
    model.add(Activation("sigmoid"))
    model.add(Dense(16, kernel_initializer="truncated_normal"))
    model.add(Activation("sigmoid"))
    model.add(Dense(8, kernel_initializer="truncated_normal"))
    model.add(Activation("sigmoid"))
    model.add(Dense(4, kernel_initializer="truncated_normal"))
    model.add(Activation("sigmoid"))
    model.add(Dense(2, kernel_initializer="truncated_normal"))
    model.add(Activation("sigmoid"))
    

    '''
    #model.add(Dropout(0.01))
    model.add(Dense(512, kernel_initializer="truncated_normal"))
    model.add(Activation("relu"))
    #model.add(Dropout(0.01))
    model.add(Dense(64, kernel_initializer="truncated_normal"))
    model.add(Activation("relu"))
    #model.add(Dropout(0.01))
    model.add(Dense(32, kernel_initializer="truncated_normal"))
    model.add(Activation("relu"))
    model.add(Dense(16, kernel_initializer="truncated_normal"))
    model.add(Activation("relu"))
    model.add(Dense(8, kernel_initializer="truncated_normal"))
    model.add(Activation("relu"))
    model.add(Dense(4, kernel_initializer="truncated_normal"))
    model.add(Activation("relu"))
    model.add(Dense(2, kernel_initializer="truncated_normal"))
    model.add(Activation("relu"))
    #model.add(SimpleRNN(150, dropout=0.1, recurrent_dropout=0.1, return_sequences=False))
    #model.add(Dense(150, activation="relu"))
    #model.add(SimpleRNN(150, dropout=0.1, recurrent_dropout=0.1, recurrent_activation="relu"))
    #model.add(Dense(150, activation="relu"))
    #model.add(SimpleRNN(150, dropout=0.1, recurrent_dropout=0.1, recurrent_activation="relu"))
    #model.add(Dense(150, activation="relu"))
    #model.add(SimpleRNN(150, dropout=0.1, recurrent_dropout=0.1, recurrent_activation="relu"))
    '''
    model.add(Dense(1, kernel_initializer="truncated_normal"))
    model.add(Activation("sigmoid"))
    

    model.summary()
    
    model.compile(optimizer=Adam(lr = 1e-5), loss='mse', metrics=['accuracy'])
    
    return model
