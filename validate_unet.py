import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

IMG_HEIGHT = 128
IMG_WIDTH = 128

folder = 'basic_testing'
samples = []
results = []

model = tf.keras.models.load_model('wrist_fracture_detection.model')

[print(i.shape, i.dtype) for i in model.inputs]
[print(o.shape, o.dtype) for o in model.outputs]
[print(l.name, l.input_shape, l.dtype) for l in model.layers]

for img in os.listdir(folder):
    img_path = os.path.join(folder, img)
    img_arr = load_img(img_path, target_size=(IMG_HEIGHT,IMG_WIDTH)) 
    # img_arr = img_to_array(img_arr)
    img_arr = np.asarray(img_arr)
    print(img_arr.shape)

    img_arr = img_arr[:,:,0]
    img_arr = np.expand_dims(img_arr , axis=0)
    print(img_arr.shape)
    # X = np.asarray(img_arr).astype('float32')
    predictions = model.predict(img_arr)
    result_arr =np.argmax(predictions, axis = 1)
    print(result_arr)

    plt.imshow(result_arr, interpolation='nearest')
    plt.show()
    # img_arr = preprocess_input(img_arr)
    # samples.append(img_arr)


# X = np.asarray(samples).astype('float32')

# predictions = model.predict(X)
# result_arr =np.argmax(predictions, axis = 1)
# print(result_arr)