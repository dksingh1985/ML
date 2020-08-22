from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_WEIGHT_PATH = 'D:\\Data\\Kaggle\\TGS\\tgs_model.r5.h5'


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myValidGen = trainGenerator(50,'D:\\Data\\Kaggle\\TGS\\train','image-test','mask-test',data_gen_args,image_color_mode = 'grayscale',save_to_dir = None)

model = unet2()

model.load_weights(MODEL_WEIGHT_PATH)

model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.0000001,momentum=0.01),metrics=["accuracy"])

#model_checkpoint = ModelCheckpoint('D:\\Data\\Kaggle\\TGS\\unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.fit_generator(myGene,steps_per_epoch=10,epochs=100000,callbacks=[model_checkpoint])

score = model.evaluate_generator(myValidGen,8,verbose=1)

print("Loss     :", score[0])
print("Accuracy :", score[1])
