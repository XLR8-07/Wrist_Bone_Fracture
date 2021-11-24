import numpy as np
import cv2
import os
import random 
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import PreTrainedModel
import xml_to_csv

TRAININGSET_DIRECTORY = r"D:\Fracture Research\dataset\fracture\training"
TRAINING_ANNOTATIONSET_DIRECTORY = r"D:\Fracture Research\dataset\fracture\training_annotations"
# PREPROCESSED_DIRECTORY = r"E:\P R O D I G Y\A C A D E M I C\C O D E\DEEP LEARNING\Face Mask Detection\preprocessed"

IMG_HEIGHT = 224
IMG_WIDTH = 224

files = []

# MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
# model = PreTrainedModel.load_model(MODEL_NAME)
# model = model.signatures['serving_default']

# for directory,_,filenames in os.walk(TRAININGSET_DIRECTORY):
#     for filename in filenames:
#         files.append(int(filename.split('.')[0]))
#         # img_path = os.path.join(directory,filename)
        
#         # img = load_img(img_path , target_size=(IMG_HEIGHT, IMG_WIDTH))
#         # plt.imshow(img)
# files = sorted(files)
# trainValFile = open(r"D:\Fracture Research\models\annotations\trainval.txt", 'w+')

# for i in files:
#     trainValFile.write(str(i)+'\n')

# trainValFile.close()
xml_to_csv.main()