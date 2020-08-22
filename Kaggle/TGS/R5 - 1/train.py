from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_SAVE_PATH = 'D:\\Data\\Kaggle\\TGS\\tgs_model.r5.hdf5'

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myTrainGen = trainGenerator(100,'D:\\Data\\Kaggle\\TGS\\train','image','mask',data_gen_args,image_color_mode = 'grayscale',save_to_dir = None)
myValidGen = trainGenerator(20,'D:\\Data\\Kaggle\\TGS\\train','image-test','mask-test',data_gen_args,image_color_mode = 'grayscale',save_to_dir = None)

#model = unet2()

#model = load_model(MODEL_SAVE_PATH)

#model.compile(optimizer = Adam(lr = 1e-5, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8), loss = 'binary_crossentropy', metrics = ['accuracy'])

#model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='loss',verbose=1, save_best_only=False,save_weights_only=False)
#model.fit_generator(myTrainGen,steps_per_epoch=5,epochs=1, verbose=1, callbacks=[model_checkpoint], validation_data = myValidGen, validation_steps = 1)

for i in range(100000000):
    print ("Epoch------------->",i,"<-------------")
    model = load_model(MODEL_SAVE_PATH)
    model.fit_generator(myTrainGen,steps_per_epoch=5,epochs=1, verbose=1, validation_data = myValidGen, validation_steps = 1)
    model.save(MODEL_SAVE_PATH)

