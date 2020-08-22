import pandas as pd
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Activation, Dense, Dropout, RepeatVector, SpatialDropout1D, GRU, SimpleRNN
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing import sequence


FILE_NAME = "NSE-SUZLON.csv"
INPUT_WINDOW_SIZE = 1
INPUT_PARAMETER = 7

s_df = pd.read_csv(FILE_NAME)

s_df = s_df.sort_index(ascending=False)

s_df = s_df.rename(columns={"Total Trade Quantity":"Volume"})


s_df["L/H"] = s_df["Low"]/s_df["High"]
s_df["H-L"] = s_df.High - s_df.Low
s_df["O-C"] = s_df.Open - s_df.Close
s_df["CO-HL_r"] = (s_df["O-C"]/s_df["H-L"])
s_df["O-L"] = s_df.Open - s_df.Low
s_df["C-L"] = s_df.Close - s_df.Low
s_df["O_r"] = s_df["O-L"]/s_df["H-L"]
s_df["C_r"] = s_df["C-L"]/s_df["H-L"]
s_df["HL_r"] = s_df["H-L"]/s_df["Low"]


s_df["Close_10ema"] = s_df["Close"].ewm(span=10, adjust=False).mean()
s_df["Close_30ema"] = s_df["Close"].ewm(span=30, adjust=False).mean()
s_df["Close_macd"] =  (s_df["Close_10ema"] - s_df["Close_30ema"])/(s_df["Close_10ema"] + s_df["Close_30ema"])

'''
s_df["Close_macd_1"] = s_df["Close_macd"]
s_df[("Close_macd")][s_df.Close_macd < 0 ] = 0
s_df.Close_macd_1[s_df.Close_macd_1 > 0 ] = 0
s_df.Close_macd_1 = -1*s_df.Close_macd_1
'''

s_df["Volume_10ema"] = s_df.Volume.ewm(span=10,adjust=False).mean()
s_df["Volume_30ema"] = s_df.Volume.ewm(span=30,adjust=False).mean()
s_df["Volume_macd"] = (s_df["Volume_10ema"] - s_df["Volume_30ema"])/ (s_df["Volume_10ema"] + s_df["Volume_30ema"])

'''
s_df["Volume_macd_1"] = s_df["Volume_macd"]
s_df["Volume_macd"][s_df["Volume_macd"] < 0] = 0
s_df["Volume_macd_1"][s_df.Volume_macd_1 > 0 ] = 0
s_df.Volume_macd_1 = -1 * s_df.Volume_macd_1
'''

rows = s_df.count()[0]

print("Data count : ", rows)

#s_df["h_mean"] = s_df["High"].rolling(30).mean()

#s_df["v_mean"] = s_df["Total Trade Quantity"].rolling(30).mean()


def get_training_data(batch_size,index=0):
    net_in = np.zeros((batch_size,INPUT_WINDOW_SIZE * INPUT_PARAMETER))
    net_out = np.zeros((batch_size,1))

    for x in range(batch_size):#,rows-1):

        #indx = index + x
        indx = random.randint(INPUT_WINDOW_SIZE, rows-batch_size)

        temp_df = s_df[indx-INPUT_WINDOW_SIZE:indx]
        temp_df = temp_df.reset_index(drop=True)
        
        out_df = s_df[indx:indx+1]
        out_df = out_df.reset_index(drop=True)

        #print("Itr:", indx, temp_df.iloc[0].Date, temp_df.iloc[29].Date, out_df.iloc[0].Date)
        
        #temp_df


        t_open = temp_df.iloc[INPUT_WINDOW_SIZE-1]["Open"]

        '''
        t_volume = temp_df.iloc[INPUT_WINDOW_SIZE-1]["Volume"]
        
        temp_df.loc[:,("Open")] = temp_df["Open"].divide(t_open)
        temp_df.loc[:,("Close")] = temp_df["Close"].divide(t_open)
        temp_df.loc[:,("High")] = temp_df["High"].divide(t_open)
        temp_df.loc[:,("Low")] = temp_df["Low"].divide(t_open)
        temp_df.loc[:,("Volume")] = temp_df["Volume"].divide(t_volume)
        '''
        
        out_df.loc[:,("Close")] = out_df["Close"].divide(t_open).subtract(1)
        
        
        lst = np.array(temp_df[["L/H","CO-HL_r","O_r","C_r","HL_r","Close_macd","Volume_macd"]])
        #lst = np.array(temp_df[["Open","High","Low","Close","Close_10ema","Close_30ema","Volume_10ema","Volume_30ema"]])
        out_lst = np.array(out_df[["Close"]])

        net_in[x] =  lst.reshape((INPUT_WINDOW_SIZE * INPUT_PARAMETER))
        net_out[x] = out_lst.reshape((1))
        #print(temp_df)
    return net_in, net_out
        

    
