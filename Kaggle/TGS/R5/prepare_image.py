from model import *
from data import *

#IMAGE_PATH = "D:\\Data\\Kaggle\\TGS\\train\\image"
#IMAGE_PATH = "D:\\Data\\Kaggle\\TGS\\train\\image-test"
IMAGE_PATH = "D:\\Data\\Kaggle\\TGS\\test\\image"

imgs = os.listdir(IMAGE_PATH)

for i in range(len(imgs)):
    print(i,">", imgs[i])
    prepareData(IMAGE_PATH,imgs[i])
