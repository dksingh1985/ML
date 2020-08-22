from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
from PIL import Image


IMAGE_SIZE = 96
ACTUAL_SIZE = 101


def adjustData(img,mask):
    img = np.divide(img,255)
    mask = np.divide(mask,255)
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    save_to_dir = None,target_size = (IMAGE_SIZE,IMAGE_SIZE),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)

    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask)
        yield (img,mask)
   
def validatorGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    save_to_dir = None,target_size = (IMAGE_SIZE,IMAGE_SIZE),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)

    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask)
        yield (img,mask)

def trainGenerator2(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    save_to_dir = None,target_size = (IMAGE_SIZE,IMAGE_SIZE),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)

    for (img,mask) in train_generator:
        temp_mask = np.zeros([mask.shape[0],2])
        img,mask = adjustData(img,mask)
        mask2 = mask.sum(1).sum(1)
        #print("mask2 : ", mask2)
        mask2[mask2 >= 0.5] = 1
        mask2[mask2 < 0.5] = 0
        total =0
        salt = 0
        for i in range(mask2.shape[0]):
            total += 1
            if (mask2[i][0] > 0):
                salt += 1
                temp_mask[i][0] = 1
                #print("Mask2:(",i ,"):", mask2[i][0])
            else:
                temp_mask[i][1] = 0
            #print("Mask2:(",i ,"):", temp_mask[i,:])
                
        print(" ")
        print("Salt = ", (salt/total), "(", salt , "/", total , ")")
        yield (img,temp_mask)
   

def testGenerator(test_path,target_size = (IMAGE_SIZE,IMAGE_SIZE)):
    files = os.listdir(test_path)
    for i in range(len(files)):
        print("F>:", os.path.join(test_path + "\\" + files[i]))
        img = Image.open(os.path.join(test_path + "\\" + files[i])).convert("L")
        #io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img.resize((IMAGE_SIZE,IMAGE_SIZE), Image.ANTIALIAS)
        img = np.divide(img,255)
        #img = trans.resize(img,target_size)
        yield img


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        #img = io.imread(item,as_gray = image_as_gray)
        img = Image.open(item).convert("L")
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        #mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = Image.open(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix)).convert("L")
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr

def imgPredictNpy(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((IMAGE_SIZE,IMAGE_SIZE), Image.ANTIALIAS)
    img = np.divide(img,255)
    img = img.reshape((1,IMAGE_SIZE,IMAGE_SIZE,1))
    return img

def mskPredictNpy(mask_path):
    img = Image.open(mask_path).convert("L")
    img = np.divide(img,255)
    img[img > 0.5] = 1
    img[img <= 0.5] = 0
    img = img.reshape((1,-1),order="F")
    return img

def saveResult(save_path,file_name,nparray):
    for i in range(nparray.shape[0]):
        img = nparray[i]
        img[img > 0.5] = 1
        img[img <= 0.5] = 0
        img = np.multiply(img,255)
        img = img.reshape((IMAGE_SIZE,IMAGE_SIZE))
        img = Image.fromarray(img.astype(np.uint8))
        img = img.resize((ACTUAL_SIZE,ACTUAL_SIZE),Image.ANTIALIAS)
        img.save(save_path + "\\" + file_name)


def conv2RLE(nparray):
    value_to_encode = 1
    rle = ""
    cont = False
    indx = 0
    for i in range(nparray.shape[1]):
        if (cont):
            if (nparray.shape[1]-1 == i):
                rle = rle + str(i-indx+1) + " "
            elif (nparray[0][i]!= value_to_encode):
                rle = rle + str(i-indx) + " "
                cont = False
        else:
            if (nparray[0][i]== value_to_encode):
                if (nparray.shape[1]-1 == i):
                    rle = rle + str(i-indx+1) + " 1"
                else:
                    rle = rle + str(i+1) + " "
                    indx = i
                    cont = True

    return rle.strip()


def RLenc(img,order='F',format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not
    
    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = [] ## list of run lengths
    r = 0     ## the current run length
    pos = 1   ## count starts from 1 per WK
    for c in bytes:
        if ( c == 0 ):
            if r != 0:
                runs.append((pos, r))
                pos+=r
                r=0
            pos+=1
        else:
            r+=1

    #if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''
    
        for rr in runs:
            z+='{} {} '.format(rr[0],rr[1])
        return z[:-1]
    else:
        return runs

