from model import *
from data import *

TEST_IMAGE_PATH = "D:\\Data\\Kaggle\\TGS\\test\\image"
PRED_MASK_PATH = "D:\\Data\\Kaggle\\TGS\\test\\mask"


model = load_model('D:\\Data\\Kaggle\\TGS\\unet_membrane.hdf5')

imgs = os.listdir(TEST_IMAGE_PATH)

for i in range(len(imgs)):
    print(i,">",TEST_IMAGE_PATH + "\\" + imgs[i])
    if (os.path.exists(PRED_MASK_PATH + "\\" + imgs[i]) == False):
        print("Converting..")
        imgNp = imgPredictNpy(TEST_IMAGE_PATH + "\\" + imgs[i])
        predict = model.predict(imgNp)
        saveResult(PRED_MASK_PATH,imgs[i],predict)
    else:
        print("Skip. Mask already exists.")
