import os

from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from model import build_Unet
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import sys

# Hyperparameters
prediction_dir = "../dataset_rokycanska/predictions/" # directory for predictions
image_dir = "../dataset_rokycanska/images/"  # directory with input RGB .jpg images
model_dir = "../models/"  # directory where trained models will be saved
model_name = "model_rokycanska_vice_trid.h5"  # name of the trained model
img_size = (160, 160)#(128, 128)               # size of images which will be processed with neural network (only width and height)


# Make and compile model
model = build_Unet((img_size+(3,)))
optimizer = tf.keras.optimizers.Adam(lr=1e-4)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics='accuracy')#optimizer="rmsprop",loss="sparse_categorical_crossentropy"
model.summary()
# save weights from model to fresh model structure
model.load_weights(model_dir+model_name)

length = len(os.listdir(image_dir))-1
for i,path in enumerate(os.listdir(image_dir)):
    # Load and plot image
    test_img = load_img(image_dir+path, target_size=img_size)
    test_img = np.asarray(test_img)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(test_img)
    ax[0].set_title('Original')

    # Load and plot original mask
    '''test_mask = load_img(masks_dir+img_name+'.png', target_size=img_size)
    test_mask = np.array(test_mask)
    test_mask -= 1
    test_mask *= 255
    ax[1].imshow(test_mask, cmap='gray')
    ax[1].set_title('Mask')'''

    # Predict and plot predicted mask
    test_img = cv2.resize(test_img, img_size)
    test_img = np.expand_dims(test_img, 0)
    result = model.predict(test_img)
    result = result[0]
    # pro pripad potreby prevest na puvodni rozmery a porovnat s originaly
    #resized = cv2.resize(result.astype('float32'), (1920, 1080), interpolation=cv2.INTER_AREA)
    ax[1].imshow(result)
    ax[1].set_title('Predicted mask')
    #plt.show()
    plt.savefig(prediction_dir+path)
    sys.stdout.write("\r" + str(i + 1) + " z " + str(length))
    sys.stdout.flush()
    plt.close(fig)