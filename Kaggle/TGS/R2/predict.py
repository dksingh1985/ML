from model import *
from data import *

TEST_IMAGE_PATH = "D:\\Data\\Kaggle\\TGS\\test\\image"
PRED_MASK_PATH = "D:\\Data\\Kaggle\\TGS\\test\\mask"



model = unet2()
model.load_weights('D:\\Data\\Kaggle\\TGS\\tgs_model.r2.h5')
model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.00001,momentum=0.01),metrics=["accuracy"])

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
