import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,EarlyStopping
from keras import backend as keras
from PIL import Image,ImageDraw
from skimage import filters,img_as_float
from skimage.io import imsave
from Model_Creation_unet import mean_iou,unet

images_train={}

for dirname, _x, filenames in os.walk(r'D:\Fracture Research\dataset\fracture\training'):
    # print(dirname,filenames)
    for filename in filenames:
      img = Image.open(os.path.join(dirname,filename)).convert('RGB')
      index = os.path.splitext(filename)[0]
      images_train[index] = img

# plt.imshow(images_train['5'])     #For testing purposes only
# plt.show()

image_train_annotations = {}

for dirname, _x, filenames in os.walk(r'D:\Fracture Research\dataset\fracture\training_annotations'):
  for filename in filenames:
    tree = ET.parse(os.path.join(dirname,filename))
    root = tree.getroot()
    index = os.path.splitext(filename)[0]
    img = images_train[index]
    width , height = img.size

    if img == None:
      del images_train[index]
      continue

    image = Image.new(mode='1',size = (width,height), color = 0)
    image1 = Image.new(mode='1' ,size = (width, height) , color = 1)

    try:
      for objects in root.iter('bndbox'):
        start1 , end1 = ( int(objects.find('xmin').text) , int(objects.find('ymin').text) )
        start2 , end2 = ( int(objects.find('xmax').text) , int(objects.find('ymax').text) )
        
        image1 = image1.crop((start1, end1 , start2, end2))
        image.paste(image1 , (start1 , end1 , start2 , end2))

        image_train_annotations[index] = image
    except AttributeError:
      del images_train[index]

# plt.imshow(image_train_annotations['4'])
# plt.show()                                #For Testing purposes only

fn = lambda x : 1 if x > 240 else 0

train_data=[images_train[i] for i in sorted(images_train)]
train_annotations=[image_train_annotations[i] for i in sorted(image_train_annotations)]

# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(train_data[0])

# plt.imshow(train_annotations[0])
# plt.show()

images_valid = {}

for dirname, _x, filenames in os.walk(r'D:\Fracture Research\dataset\fracture\validation'):
  for filename in filenames:
    img = Image.open(os.path.join(dirname,filename)).convert('RGB')
    index = os.path.splitext(filename)[0]
    images_valid[index] = img

image_valid_annotations = {}

for dirname, _x, filenames in os.walk(r'D:\Fracture Research\dataset\fracture\validation_annotations'):
  for filename in filenames:
    tree = ET.parse(os.path.join(dirname,filename))
    root = tree.getroot()
    index = os.path.splitext(filename)[0]
    img = images_valid[index]
    width , height = img.size

    if img == None:
      del images_valid[index]
      continue
    
    image = Image.new(mode = '1' , size = ( width, height ) , color = 0)
    image1 = Image.new(mode = '1' , size = ( width, height ) , color = 1)

    try:
      for objects in root.iter('bndbox'):
        start1 , end1 = ( int(objects.find('xmin').text) , int(objects.find('ymin').text) )
        start2 , end2 = ( int(objects.find('xmax').text) , int(objects.find('ymax').text) )
        
        image1 = image1.crop((start1, end1 , start2, end2))
        image.paste(image1 , (start1 , end1 , start2 , end2))
      
      image_valid_annotations[index] = image
    except AttributeError:
      del images_valid[index]

fn = lambda x : 1 if x > 240 else 0

valid_data = [images_valid[i] for i in sorted(images_valid)]
valid_annotations = [image_valid_annotations[i] for i in sorted(image_valid_annotations)] 