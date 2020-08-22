from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_WEIGHT_PATH = 'D:\\Data\\Kaggle\\TGS\\tgs_model.r5.h5'
MODEL_SAVE_PATH = 'D:\\Data\\Kaggle\\TGS\\tgs_model.r5.hdf5'

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myTrainGen = trainGenerator(75,'D:\\Data\\Kaggle\\TGS\\train','image','mask',data_gen_args,image_color_mode = 'grayscale',save_to_dir = None)
myValidGen = trainGenerator(20,'D:\\Data\\Kaggle\\TGS\\train','image-test','mask-test',data_gen_args,image_color_mode = 'grayscale',save_to_dir = None)

#model = unet2()
model = load_model(MODEL_SAVE_PATH)
model.summary()
    
#model.load_weights(MODEL_WEIGHT_PATH)



for i in range(1000000000):
    #model = load_model(MODEL_SAVE_PATH)
    model.fit_generator(myTrainGen,steps_per_epoch=5,epochs=1, verbose=1, validation_data = myValidGen, validation_steps = 1)
    model.save(MODEL_SAVE_PATH, overwrite=True)

