import pandas as pd
import os


df = pd.read_csv("D:\\Data\\Kaggle\\Airbus\\train_ship_segmentations.csv")
imgs = os.listdir("D:\\Data\\Kaggle\\Airbus\\train_v2")

for i in range(len(imgs)):
    try:
        if (len(df[df.ImageId == imgs[i]]) == 0):
            os.rename("D:\\Data\\Kaggle\\Airbus\\train_v2\\" + imgs[i],"D:\\Data\\Kaggle\\Airbus\\train_vd\\" + imgs[i])
            print(i, ">" , imgs[i])
        else:
            print(i)
    except FileNotFoundError:
        print("Error", i, ">" , imgs[i])
        #os.rename("D:\\Data\\Kaggle\\Airbus\\train_v2\\" + imgs[i],"D:\\Data\\Kaggle\\Airbus\\train_vd\\" + imgs[i])
