from model import *
from data import *

BATCH_SIZE = 200

#model = get_feature_map()

#model = load_model(MODEL_SAVE_PATH)

imgs = os.listdir(IMAGE_FOLDER)
random.shuffle(imgs)
print("Total image count", len(imgs))
for i in range(len(imgs)):
    img = get_image_np(imgs[i])
    img_gry = get_image_gray_np(imgs[i])

    img_new = img_gry.filter(ImageFilter.EDGE_ENHANCE)#.GaussianBlur(2))
    #img_new = img_new.filter(ImageFilter.EDGE_ENHANCE)
    img_new = otsu(img_new)

    img_new1 = img_new.copy()
    print("Pixcel:", img_new.sum(), " > ", 75202560)
    if (img_new.sum() > 75202560):
        img_new1[img_new1 == 255] = -255
        img_new1[True] += 255

    img_new2 = compact_pixel(img_new1)
    img_new2 = compact_pixel(img_new2)
    img_new2 = expend_pixel(img_new2)
    img_new2 = expend_pixel(img_new2)
    
    #img_new1 = img_gry.filter(ImageFilter.GaussianBlur(2))
    #img_new1 = img_new1.filter(ImageFilter.EDGE_ENHANCE)
    #img_new1 = otsu(img_new)
    
    #img_new = img_gry.filter(ImageFilter.BLUR)
    #img_new = img_gry.filter(ImageFilter.EDGE_ENHANCE)
    #img_new = img_gry.filter(ImageFilter.FIND_EDGES)
    # = img_gry.histogram()
    #a = otsu(img_gry)
    
    '''
    if (os.path.exists(MASK_FOLDER + "\\" + imgs[i]) == False):
            generate_mask(imgs[i])
    msk = get_mask_encode(get_mask_np(imgs[i]))
    predict = model.predict(img.reshape((1,768,768,3)))

    msk = np.multiply(msk,255)
    msk = msk.reshape(32,32)

    predict = np.multiply(predict,255)
    predict = predict.reshape(32,32)
    '''
    
    fig, ax = plt.subplots(1,2)
    ax = ax.flatten()
    im = ax[0].imshow(img)
    im = ax[1].imshow(img_new2,cmap='gray')
    #im = ax[2].imshow(img_new1,cmap='gray')
    #im = ax[3].imshow(img_new2,cmap='gray')
    '''
    print("a :", len(a))
    for i in range(len(a)):
        im = ax[2].bar(i,a[i])
    '''
    
    plt.show()

