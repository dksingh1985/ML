import pandas as pd
import numpy as np
import os

path = "D:\\Data\\Kaggle\\CIFAR-10\\"

trainLabel = pd.read_csv("D:\\Data\\Kaggle\\CIFAR-10\\trainLabels.csv")

##trainLabel = np.array(trainLabel)
##
##trainLabel[ trainLabel == "airplane" ] = 0
##trainLabel[ trainLabel == "automobile" ] = 1
##trainLabel[ trainLabel == "bird" ] = 2
##trainLabel[ trainLabel == "cat" ] = 3
##trainLabel[ trainLabel == "deer" ] = 4
##trainLabel[ trainLabel == "dog" ] = 5
##trainLabel[ trainLabel == "frog" ] = 6
##trainLabel[ trainLabel == "horse" ] = 7
##trainLabel[ trainLabel == "ship" ] = 8
##trainLabel[ trainLabel == "truck" ] = 9

if not os.path.exists(path + "img\\airplane"):
    os.mkdir(path + "img\\airplane")
if not os.path.exists(path + "img\\automobile"):
    os.mkdir(path + "img\\automobile")
if not os.path.exists(path + "img\\bird"):
    os.mkdir(path + "img\\bird")
if not os.path.exists(path + "img\\cat"):
    os.mkdir(path + "img\\cat")
if not os.path.exists(path + "img\\deer"):
    os.mkdir(path + "img\\deer")
if not os.path.exists(path + "img\\dog"):
    os.mkdir(path + "img\\dog")
if not os.path.exists(path + "img\\frog"):
    os.mkdir(path + "img\\frog")
if not os.path.exists(path + "img\\horse"):
    os.mkdir(path + "img\\horse")
if not os.path.exists(path + "img\\ship"):
    os.mkdir(path + "img\\ship")
if not os.path.exists(path + "img\\truck"):
    os.mkdir(path + "img\\truck")


for index,row in trainLabel.iterrows():
    print(row["id"],row["label"])
    if os.path.exists(path + "train\\" + str(row["id"]) + ".png" ):
        print(".....")
        os.rename(path + "train\\" + str(row["id"]) + ".png" , path + "img\\" + row["label"] + "\\" + str(row["id"]) + ".png")
    






