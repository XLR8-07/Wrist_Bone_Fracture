import tensorflow as tf 
from Model_Creation_unet import unet
import matplotlib.pyplot as plt
import numpy as np
from dataset_augmentation_unet import train_generator
# gpus = tf.config.list_physical_devices('GPU') 
# print(gpus)

with tf.device('/gpu:0'):
    model = unet()
    Base = model.compile(optimizer='adam', loss="binary_crossentropy" , metrics=["accuracy"]) 
    results = model.fit(train_generator , epochs=40 , steps_per_epoch=100)
    print("[INFO] Saving the Model.....")
    model.save("wrist_fracture_detection.model",save_format="h5") 

N = 40
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N),results.history["loss"] , label="train_loss")
# plt.plot(np.arange(0,N),results.history["val_loss"] , label="val_loss")
plt.plot(np.arange(0,N),results.history["accuracy"] , label="train_acc")
# plt.plot(np.arange(0,N),results.history["val_accuracy"] , label="val_acc")
plt.title("training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("accuracy_Graph.png")
    
# if gpus:
#   try: 
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU') 
#     model = unet() 
#     model.compile(optimizer='adam', loss='binary_crossentropy' ,metrics=['binary_accuracy']) 
#     # earlystopper = EarlyStopping(patience=5, verbose=10) 
#     # checkpointer = ModelCheckpoint('/content/drive/MyDrive/Fracture Research/modelunet.h5', verbose=1, save_best_only=True) 
#     # results = model.fit_generator(train_generator, epochs=5, callbacks=[earlystopper, checkpointer]) 
#   except RuntimeError as e:
#     print(e)