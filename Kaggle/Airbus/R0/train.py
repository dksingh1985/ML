from model import *
from data import *

BATCH_SIZE = 50

model = get_feature_map()

#model = load_model(MODEL_SAVE_PATH)

for i in range(1000000000):
    print("Epoch : ", i, "Trained on image :" , i * BATCH_SIZE)
    image, mask = get_training_date(BATCH_SIZE)
    print(image.shape,mask.shape)
    model.fit(image , mask , batch_size=5, epochs=1, verbose=1)
    model.save(MODEL_SAVE_PATH)



