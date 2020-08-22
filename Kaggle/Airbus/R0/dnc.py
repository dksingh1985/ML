import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd



IMAGE_PATH = "D:\\Data\\Kaggle\\Airbus\\train\\image\\008d2170e.jpg"
GRID_SIZE = 24
MASK_DF = pd.read_csv("D:\\Data\\Kaggle\\Airbus\\train_ship_segmentations.csv")

img = Image.open(IMAGE_PATH)#.convert("L")
img = np.array(img)

temp_img = np.zeros(589824, order="F")


for i in range(768//GRID_SIZE):
    img[(i*GRID_SIZE):(i*GRID_SIZE) + 1,:,:] = 0
    img[:,(i*GRID_SIZE):(i*GRID_SIZE) + 1,:] = 0
    for j in range(768//GRID_SIZE):
        avg = np.histogram(img[(i*GRID_SIZE):((i+1)*GRID_SIZE),(j*GRID_SIZE):((j+1)*GRID_SIZE)])
        #print(i,j,avg)
        
'''
temp_df = MASK_DF[MASK_DF.ImageId == "008d2170e.jpg"]

if (len(temp_df) > 0):
    for i in range(len(temp_df)):
        encoded_str = str(temp_df.iloc[i].EncodedPixels)
        print(encoded_str)
        encoded_list = encoded_str.split()
        print("encoded_list length : ",len(encoded_list))
        for j in range(len(encoded_list) // 2):
            for k in range(int(encoded_list[j*2])- 1,int(encoded_list[j*2]) + int(encoded_list[(j*2)+1]) - 1):
                temp_img[k] = 1 
    
#img = temp_img.reshape((768,768),order="F")
'''
fig, ax = plt.subplots()
im = ax.imshow(img,cmap='gray')
ax.axis('on')
plt.show()
