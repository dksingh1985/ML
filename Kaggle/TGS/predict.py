from model import *
from data import *

TEST_IMAGE_PATH = "D:\\Data\\Kaggle\\TGS\\test\\image"
PRED_MASK_PATH = "D:\\Data\\Kaggle\\TGS\\test\\mask"
MODEL_SAVE_PATH = 'D:\\Data\\Kaggle\\TGS\\tgs_model.r5.hdf5'
BATCH_SIZE = 360


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

model = load_model(MODEL_SAVE_PATH)


imgs = os.listdir(TEST_IMAGE_PATH)

for i in range(len(imgs) // BATCH_SIZE):
    print("Batch no  >",i,"/", (len(imgs) // BATCH_SIZE))
    imgNp = imgPredictNpy2(TEST_IMAGE_PATH, imgs, i, BATCH_SIZE)
    predict = model.predict(imgNp)
    saveResult2(PRED_MASK_PATH,imgs,predict,i,BATCH_SIZE)
    
