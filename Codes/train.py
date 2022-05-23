import os
import cv2
import matplotlib.pyplot as plt
from model import build_Unet
from tensorflow import keras
from DatasetGenerator import DataGen
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import random


if __name__=="__main__":
    # Hyperparameters
    image_dir = "../dataset_rokycanska/images/"  # directory with input RGB .jpg images
    masks_dir = "../dataset_rokycanska/masks/"  # directory with input .png images with values 1,2,3,4....
    model_dir = "../models/"  # directory where trained models will be saved
    model_name = "model_rokycanska_vice_trid.h5"       # name of the trained model
    img_size = (160, 160) #(128, 128)           # size of images which will be processed with neural network (only width and height)
    num_classes = 4                 # number of classes included in .png masks
    batch_size = 1                 # batch size of input sets
    epochs = 100                     # number of epochs

    # Sorting paths to images and masks to allign them into two arrays of paths
    input_img_paths = sorted(
        [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.endswith(".jpg")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(masks_dir, fname)
            for fname in os.listdir(masks_dir)
            if fname.endswith(".png")
        ]
    )
    # Printing evidence of readed files
    print(f"Found\n{len(input_img_paths)} images\n{len(target_img_paths)} masks")
    print("Printing few examples of aligned and sorted paths...")
    nOfExamples = 10
    for input_path, target_path in zip(input_img_paths[:nOfExamples], target_img_paths[:nOfExamples]):
        print(input_path, "|", target_path)

    # Show image with mask of two classes
    test_no = 0
    test_image = np.asarray(load_img(input_img_paths[test_no]))
    test_mask = np.asarray(load_img(target_img_paths[test_no], color_mode="grayscale"))
    print(f"values in mask: {np.unique(test_mask)}")
    # plotting
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(test_image)
    ax[0].set_title('Original')
    ax[1].imshow(test_mask)
    ax[1].set_title('Mask')
    plt.show()

    # Split img paths into a training and a validation set
    val_samples = 5
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    # Making data sequence for model training
    train_gen = DataGen(batch_size=batch_size, img_size=img_size, input_img_paths=train_input_img_paths, target_img_paths=train_target_img_paths)
    val_gen = DataGen(batch_size=batch_size, img_size=img_size, input_img_paths=val_input_img_paths, target_img_paths=val_target_img_paths)
    # printing how many images is in each sequence
    print(f"validation samples {val_samples}\ntrain input {len(train_input_img_paths)}\nvalidation {len(val_input_img_paths)}\n"
          f"val_gen {val_gen.__len__()}\ntrain_gen {train_gen.__len__()}")

    # Build model and comppile
    model = build_Unet((img_size+(3,)), model_name=model_name)
    optimizer = tf.keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics='accuracy')#optimizer="rmsprop",loss="sparse_categorical_crossentropy"
    model.summary()
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_dir+'best_checkpoint_'+model_name, save_best_only=True)
    ]

    # Train the model
    model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
    # Save model
    model.save(model_dir+model_name)


    # show final prediction of model after training
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(test_image)
    ax[0].set_title('Original')
    ax[1].imshow(test_mask)
    ax[1].set_title('Mask')
    test_image = cv2.resize(test_image, img_size)
    test_image = np.expand_dims(test_image, 0)
    result = model.predict(test_image)
    #print(np.histogram(result[0]))
    result = result[0]
    resized = cv2.resize(result.astype('float32'), (1920, 1080), interpolation=cv2.INTER_AREA)
    ax[2].imshow(resized)
    ax[2].set_title('Predicted mask')
    plt.show()