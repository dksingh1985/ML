from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_WEIGHT_PATH = 'D:\\Data\\Kaggle\\TGS\\tgs_model.r4.h5'

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myTrainGen = trainGenerator(50,'D:\\Data\\Kaggle\\TGS\\train','image','mask',data_gen_args,image_color_mode = 'grayscale',save_to_dir = None)
myValidGen = trainGenerator(5,'D:\\Data\\Kaggle\\TGS\\train','image-test','mask-test',data_gen_args,image_color_mode = 'grayscale',save_to_dir = None)

model = unet2()

model.load_weights(MODEL_WEIGHT_PATH)

#model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.001,momentum=0.1),metrics=["accuracy"])
model.compile(optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8), loss = 'binary_crossentropy', metrics = ['accuracy'])

model_checkpoint = ModelCheckpoint(MODEL_WEIGHT_PATH, monitor='loss',verbose=1, save_best_only=False,save_weights_only=True)

model.fit_generator(myTrainGen,steps_per_epoch=10,epochs=100000000, verbose=1, callbacks=[model_checkpoint], validation_data = myValidGen, validation_steps = 1)

#testGene = testGenerator("D:\\Data\\Kaggle\\TGS\\train\\test")
#results = model.evaluate_generator(myTrainGen,0,verbose=1)
#saveResult("D:\\Data\\Kaggle\\TGS\\train\\test",results)
