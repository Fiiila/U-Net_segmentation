from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from model import build_Unet
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Hyperparameters
image_dir = "../dataset_rokycanska/images/"  #"dataset/images/"       # directory with input RGB .jpg images
masks_dir = "../dataset_rokycanska/masks/"  #"dataset/masks/"        # directory with input .png images with values 1,2,3,4....
model_dir = "../models/"  # directory where trained models will be saved
model_name = "model_rokycanska_auta.h5"#"model_1.h5"  # name of the trained model
img_size = (160, 160)#(128, 128)               # size of images which will be processed with neural network (only width and height)
img_name = '000'                    # name of image to test predictions

# Make and compile model
model = build_Unet((img_size+(3,)))
optimizer = tf.keras.optimizers.Adam(lr=1e-4)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics='accuracy')#optimizer="rmsprop",loss="sparse_categorical_crossentropy"
model.summary()
# save weights from model to fresh model structure
model.load_weights(model_dir+model_name)

# Load and plot image
test_img = load_img(image_dir+img_name+'.jpg', target_size=img_size)
test_img = np.array(test_img)
fig, ax = plt.subplots(1, 3)
ax[0].imshow(test_img)
ax[0].set_title('Original')

# Load and plot original mask
test_mask = load_img(masks_dir+img_name+'.png', target_size=img_size)
test_mask = np.array(test_mask)
test_mask -= 1
test_mask *= 255
ax[1].imshow(test_mask)
ax[1].set_title('Mask')

# Predict and plot predicted mask
test_img = cv2.resize(test_img, img_size)
test_img = np.expand_dims(test_img, 0)
result = model.predict(test_img)
result = result[0]
# pro pripad potreby prevest na puvodni rozmery a porovnat s originaly
#resized = cv2.resize(result.astype('float32'), (1920, 1080), interpolation=cv2.INTER_AREA)
ax[2].imshow(result)
ax[2].set_title('Predicted mask')
plt.show()
fig.savefig("dataset_preparation/dataset/predictions_original_masks/"+img_name+".jpg")