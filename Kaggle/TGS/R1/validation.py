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

myGene = validatorGenerator(50,'D:\\Data\\Kaggle\\TGS\\train','image-test','mask-test',data_gen_args,save_to_dir = None)

#model = unet2()
model = load_model('D:\\Data\\Kaggle\\TGS\\unet_membrane.hdf5')
#model_checkpoint = ModelCheckpoint('D:\\Data\\Kaggle\\TGS\\unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.fit_generator(myGene,steps_per_epoch=10,epochs=100000,callbacks=[model_checkpoint])

score = model.evaluate_generator(myGene,400,verbose=1)

for i in range(score.shape[0]):
    print("Score " , i , " :", score[i])


print("Loss     :", score[0])
print("Accuracy :", score[1])
