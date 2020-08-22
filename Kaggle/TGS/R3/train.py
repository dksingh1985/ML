from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(20,'D:\\Data\\Kaggle\\TGS\\train','image','mask',data_gen_args,image_color_mode = 'rgb',save_to_dir = None)

#model = unet2()

model = load_model('D:\\Data\\Kaggle\\TGS\\tgs_model.r3.hdf5')


#model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.1,momentum=0.01),metrics=["accuracy"])
#model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
#model = load_weights('D:\\Data\\Kaggle\\TGS\\tgs_model.r3.hdf5')

model_checkpoint = ModelCheckpoint('D:\\Data\\Kaggle\\TGS\\tgs_model.r3.hdf5', monitor='loss',verbose=1, save_best_only=False)
model.fit_generator(myGene,steps_per_epoch=20,epochs=100000,callbacks=[model_checkpoint])

#testGene = testGenerator("D:\\Data\\Kaggle\\TGS\\train\\test")
#results = model.evaluate_generator(myGene,0,verbose=1)
#saveResult("D:\\Data\\Kaggle\\TGS\\train\\test",results)
