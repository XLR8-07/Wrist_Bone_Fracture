import keras.preprocessing.image as image 
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters,img_as_float
from PreProcessing_unet import train_annotations,valid_annotations,valid_data,train_data

image_data_gen = image.ImageDataGenerator(rotation_range = 40, horizontal_flip = True, vertical_flip = True, zoom_range = 0.2, shear_range = 0.2,width_shift_range = 0.2, height_shift_range = 0.2) 
mask_data_gen = image.ImageDataGenerator(rotation_range = 40, horizontal_flip = True, vertical_flip = True, zoom_range = 0.2, shear_range = 0.2,width_shift_range = 0.2, height_shift_range = 0.2) 
validimg_data_gen = image.ImageDataGenerator() 
validmask_data_gen = image.ImageDataGenerator()

current = r'D:\Fracture Research\present'
if(not os.path.isdir(os.path.join(current,'valid_images'))):
    os.mkdir(os.path.join(current,'valid_images'))
    os.mkdir(os.path.join(current,'valid_annotations'))
    os.mkdir(os.path.join(current,'train_annotations'))
    os.mkdir(os.path.join(current,'train_images'))
    os.mkdir(os.path.join(current,'valid_images/present'))
    os.mkdir(os.path.join(current,'valid_annotations/present'))
    os.mkdir(os.path.join(current,'train_images/present'))
    os.mkdir(os.path.join(current,'train_annotations/present'))

for dirname,_,filenames in os.walk(os.path.join(current,'train_images/present')):
    for file in filenames:
        os.remove(os.path.join(dirname,file))

for dirname,_,filenames in os.walk(os.path.join(current,'train_annotations/present')):
    for file in filenames:
        os.remove(os.path.join(dirname,file))
        
for dirname,_,filenames in os.walk(os.path.join(current,'valid_images/present')):
    for file in filenames:
        os.remove(os.path.join(dirname,file))
#os.mkdir(os.path.join(current,'train_images'))
for dirname,_,filenames in os.walk(os.path.join(current,'valid_annotations/present')):
    for file in filenames:
        os.remove(os.path.join(dirname,file))

from imageio import imwrite

train_dir=os.path.join(current,'train_images/present')
train_an_dir=os.path.join(current,'train_annotations/present')
valid_dir=os.path.join(current,'valid_images/present')
valid_an_dir=os.path.join(current,'valid_annotations/present')
fn = lambda x : 1 if x > 240 else 0
for i,v in enumerate(train_data):
    if v is not None:
        cv2.imwrite(os.path.join(train_dir,str(i)+'.png'),img_as_float(np.float32(v)))

#print(type())
for i,v in enumerate(train_annotations):
    print(v)
    if v is not None:
        val = filters.threshold_otsu(np.float32(v))
        mask = np.float32(v)< val
        plt.imshow(img_as_float(mask))
        #print(img_as_float(mask))
        cv2.imwrite(os.path.join(train_an_dir,str(i)+'.png'),img_as_float(mask))
for i,v in enumerate(valid_data):
    if v is not None:
        cv2.imwrite(os.path.join(valid_dir,str(i)+'.png'),img_as_float(np.float32(v)))

for i,v in enumerate(valid_annotations):
    print(v)
    if v is not None:
        val = filters.threshold_otsu(np.float32(v))
        mask = np.float32(v)< val
        plt.imshow(mask)
        cv2.imwrite(os.path.join(valid_an_dir,str(i)+'.png'),img_as_float(mask))

image_array_gen = image_data_gen.flow_from_directory(directory=r'D:\Fracture Research\present\train_images', class_mode = None, target_size = (128,128),color_mode='grayscale',batch_size=10) 
mask_array_gen = mask_data_gen.flow_from_directory(directory=r'D:\Fracture Research\present\train_annotations', class_mode = None, target_size = (128,128),color_mode='grayscale',batch_size=10) 
valid_image_array_gen = validimg_data_gen.flow_from_directory(directory= r'D:\Fracture Research\present\valid_images', class_mode = None, target_size = (128,128),color_mode='grayscale',batch_size=10) 
valid_mask_array_gen = validmask_data_gen.flow_from_directory(directory= r'D:\Fracture Research\present\valid_annotations', class_mode = None, target_size = (128,128),color_mode='grayscale',batch_size=10) 
train_generator = zip(image_array_gen, mask_array_gen) 
valid_generator = zip(valid_image_array_gen, valid_mask_array_gen)