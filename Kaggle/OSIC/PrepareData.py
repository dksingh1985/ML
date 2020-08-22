import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import os
import re
from PIL import Image

import osci_model



# Create base director for Train .dcm files
director = "D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\png"


df = pd.read_csv("D:\\Data\\OSIC\\osic-pulmonary-fibrosis-progression\\train.csv")

p_lst = df['Patient'].unique().tolist()

x = 0
p_count = 5
t_data_X = np.zeros((p_count,1024,512,512,1))
m = osci_model.get_model(512)
while(x<1):
    x = x+1
    p_lst_r = random.choices(p_lst,k=p_count)
    t_data_X = np.zeros((p_count,1024,512,512,1))
    t_data_Y = np.zeros((p_count,140))
    #print(p_lst_r)
    for i in range(p_count):
        path = director + "\\" + p_lst_r[i]
        #print(x,"----",i,"-----",p_lst_r[i]) 
        print(x,"----",i,"-----",path)

        files = []
        for png in list(os.listdir(path)):
            files.append(png)
        files.sort(key=lambda f: int(re.sub('\D', '', f)))

        j = 0
        for png in files:
            #print("------->", png, end="\r")
            img = Image.open(path + "\\" + png).convert('L')
            img = np.array(img).reshape(512,512,1)
            img = img /255
            t_data_X[i][j] = img
            j = j + 1
    print('##################################### Predict : ')
    yhat = m.predict(t_data_X)
    print(yhat)
            
            
