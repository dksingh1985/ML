from config import *



MASK_DF = pd.read_csv("D:\\Data\\Kaggle\\Airbus\\train_ship_segmentations_v2.csv")

def generate_mask(image_name):

    print("Creating mask for image id : ", image_name)
    mask = np.zeros(589824, order="F")

    '''
    for i in range(768//GRID_SIZE):
        for j in range(768//GRID_SIZE):
            avg = np.histogram(img[(i*GRID_SIZE):((i+1)*GRID_SIZE),(j*GRID_SIZE):((j+1)*GRID_SIZE)])
            print(i,j,avg)
    '''

    temp_df = MASK_DF[MASK_DF.ImageId == image_name]

    if (len(temp_df) > 0):
        for i in range(len(temp_df)):
            encoded_str = str(temp_df.iloc[i].EncodedPixels)
            #print(encoded_str)
            encoded_list = encoded_str.split()
            #print("encoded_list length : ",len(encoded_list))
            for j in range(len(encoded_list) // 2):
                for k in range(int(encoded_list[j*2])- 1,int(encoded_list[j*2]) + int(encoded_list[(j*2)+1]) - 1):
                    mask[k] = 255 
        
    img = mask.reshape((768,768),order="F")
    img = Image.fromarray(img)
    img = img.convert("L")
    img.save(MASK_FOLDER + "\\" + image_name)


def generate_mask_all():
    imgs = os.listdir(IMAGE_FOLDER)
    print("Total image count", len(imgs))
    for i in range(len(imgs)):
        if (os.path.exists(MASK_FOLDER + "\\" + imgs[i]) == False):
            generate_mask(imgs[i])


def get_image_np(image_name):
    #imgs = np.zeros((batch_size,IMAGE_SIZE,IMAGE_SIZE,3))
    img = Image.open(IMAGE_FOLDER + "\\" + image_name)
    #img = img.resize((IMAGE_SIZE,IMAGE_SIZE), Image.ANTIALIAS)
    img = np.divide(img,255)
    #img = img.reshape((IMAGE_SIZE,IMAGE_SIZE,3))
    return img

def get_image_gray_np(image_name):
    #imgs = np.zeros((batch_size,IMAGE_SIZE,IMAGE_SIZE,3))
    img = Image.open(IMAGE_FOLDER + "\\" + image_name)
    img = img.convert("L")
    #img = img.resize((IMAGE_SIZE,IMAGE_SIZE), Image.ANTIALIAS)
    #img = np.divide(img,255)
    #img = img.reshape((IMAGE_SIZE,IMAGE_SIZE,3))
    return img
    
def get_mask_np(mask_name):
    #msk = np.zeros((batch_size,IMAGE_SIZE,IMAGE_SIZE,3))
    msk = Image.open(MASK_FOLDER + "\\" + mask_name).convert("L")
    #msk = np.array(msk)
    msk = np.divide(msk,255)
    #msk[msk >= 1] = 1
    #msk[msk < 1] = 0
    return msk

def get_mask_encode(mask_np):
    msk = np.zeros((32,32))
    for j in range(IMAGE_SIZE//GRID_SIZE):
        for k in range(IMAGE_SIZE//GRID_SIZE):
            temp_msk = mask_np[(j*GRID_SIZE):((j+1)*GRID_SIZE),(k*GRID_SIZE):((k+1)*GRID_SIZE)]
            #print(temp_msk.shape)
            msk[j,k] = np.sum(temp_msk)
    msk[msk >= 1] = 1
    msk[msk < 1] = 0
    msk = msk.reshape((1024), order="F")
    return msk
    
def get_training_date(batch_size):
    imgList = os.listdir(IMAGE_FOLDER)
    imgList =  random.sample(imgList,batch_size)
    random.shuffle(imgList)
    imgs = np.zeros((batch_size,IMAGE_SIZE,IMAGE_SIZE,3))
    msks = np.zeros((batch_size,1024))
    for i in range(batch_size):
        imgs[i] = get_image_np(imgList[i])
   
    for i in range(batch_size):
        if (os.path.exists(MASK_FOLDER + "\\" + imgList[i]) == False):
            generate_mask(imgList[i])

        msk = get_mask_np(imgList[i])
        msks[i] =   get_mask_encode(msk)  
    return imgs,msks

def otsu(image):
    gray = np.array(image)
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weigth = 1.0/pixel_number
    his, bins = np.histogram(gray, np.array(range(0, 256)))
    final_thresh = -1
    final_value = -1
    for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        Wb = np.sum(his[:t]) * mean_weigth
        Wf = np.sum(his[t:]) * mean_weigth

        mub = np.mean(his[:t])
        muf = np.mean(his[t:])

        value = Wb * Wf * (mub - muf) ** 2

        #print("Wb", Wb, "Wf", Wf)
        #print("t", t, "value", value,"final_thresh",final_thresh, "final_value", final_value)

        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = gray.copy()
    print(final_thresh)
    #print(final_img)
    final_img[final_img >= final_thresh] = 255
    final_img[final_img < final_thresh] = 0
    return final_img

def compact_pixel(img):
    temp_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i,j] == 0):
                if (i < IMAGE_SIZE-1 ):
                    temp_img[i+1,j] = 0
                if (i > 0):
                    temp_img[i-1,j] = 0
                if (j < IMAGE_SIZE-1):
                    temp_img[i,j+1] = 0
                if (j > 0):
                    temp_img[i,j-1] = 0
    return temp_img

def expend_pixel(img):
    temp_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i,j] == 255):
                if (i < IMAGE_SIZE-1 ):
                    temp_img[i+1,j] = 255
                if (i > 0):
                    temp_img[i-1,j] = 255
                if (j < IMAGE_SIZE-1):
                    temp_img[i,j+1] = 255
                if (j > 0):
                    temp_img[i,j-1] = 255
    return temp_img

def imgHist(image):
    gray = np.array(image)
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weigth = 1.0/pixel_number
    his, bins = np.histogram(gray, np.array(range(0, 256)))
    final_thresh = -1
    final_value = -1
    for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        Wb = np.sum(his[:t]) * mean_weigth
        Wf = np.sum(his[t:]) * mean_weigth

        mub = np.mean(his[:t])
        muf = np.mean(his[t:])

        value = Wb * Wf * (mub - muf) ** 2

        #print("Wb", Wb, "Wf", Wf)
        #print("t", t, "value", value,"final_thresh",final_thresh, "final_value", final_value)

        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = gray.copy()
    print(final_thresh)
    #print(final_img)
    final_img[final_img > final_thresh] = 255
    final_img[final_img < final_thresh] = 0
    return final_img
