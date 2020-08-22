import os
import numpy as np
import pandas as pd
import random
'''
#for fn in ml:
#    print(fn)

try:
    os.rmdir("D:\\Data\\Kaggle\\TGS\\train\\image")
except FileNotFoundError:
    print("error: T")

try:
    os.rmdir("D:\\Data\\Kaggle\\TGS\\train\\image-test")
except FileNotFoundError:
    print("error: T")

try:
    os.rmdir("D:\\Data\\Kaggle\\TGS\\train\\mask")
except FileNotFoundError:
    print("error: C")

try:
    os.rmdir("D:\\Data\\Kaggle\\TGS\\train\\mask-test")
except FileNotFoundError:
    print("error: C")

os.mkdir("D:\\Data\\Kaggle\\TGS\\train\\image")
os.mkdir("D:\\Data\\Kaggle\\TGS\\train\\mask")
os.mkdir("D:\\Data\\Kaggle\\TGS\\train\\image-test")
os.mkdir("D:\\Data\\Kaggle\\TGS\\train\\mask-test")
'''

ml = os.listdir("D:\\Data\\Kaggle\\TGS\\train\\image-test")
random.shuffle(ml)
v_ml = random.sample(ml,400)

for fn in v_ml:
    print(fn)
    os.rename("D:\\Data\\Kaggle\\TGS\\train\\image-test\\" + fn, "D:\\Data\\Kaggle\\TGS\\train\\image\\" + fn)
    os.rename("D:\\Data\\Kaggle\\TGS\\train\\mask-test\\" + fn, "D:\\Data\\Kaggle\\TGS\\train\\mask\\" + fn)
'''
ml = os.listdir("D:\\Data\\Kaggle\\TGS\\train\\images")

for fn in ml:
    print(fn)
    os.rename("D:\\Data\\Kaggle\\TGS\\train\\images\\" + fn, "D:\\Data\\Kaggle\\TGS\\train\\image\\" + fn)
    os.rename("D:\\Data\\Kaggle\\TGS\\train\\masks\\" + fn, "D:\\Data\\Kaggle\\TGS\\train\\mask\\" + fn)
'''
