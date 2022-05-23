import sys

import cv2
import os

import matplotlib.pyplot as plt
import numpy as np


def brighten_img(image_path,treshold=100):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = np.where(image>treshold,image,255)
    return image

def brighten_all(path,save_path):
    length = len(os.listdir(path))
    for i,dirname in enumerate(os.listdir(path)):
        image = brighten_img(path+dirname)
        cv2.imwrite(save_path+dirname.replace("jpg", "png"), image)
        sys.stdout.write("\r"+str(i+1) + " z " + str(length))
        sys.stdout.flush()
    print('\nDone')

def rename_images_in_dir(dir):
    for i,dirname in enumerate(os.listdir(dir)):
        os.rename(f"{dir}{dirname}", f"{dir}{i:03d}.jpg")
        print(f"{dir}{dirname} --> renamed to --> {dir}{i:03d}.jpg")
    print('done')

def make_mask(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image > 50
    image = image+1
    return image

def make_masks(path, save_path):
    length = len(os.listdir(path))
    for i, dirname in enumerate(os.listdir(path)):
        image = make_mask(path + dirname)
        cv2.imwrite(save_path + dirname.replace("jpg", "png"), image)
        sys.stdout.write("\r" + str(i + 1) + " z " + str(length))
        sys.stdout.flush()
    print('\nDone')

if __name__=="__main__":
    path_to_images = '../dataset_preparation/dataset/source/'
    #path_to_masks = 'dataset/mask_phase1'
    #rename_images_in_dir(path_to_images)
    #brighten_all(path_to_images, 'dataset/mask_phase1/')
    #make_masks("dataset/mask_phase2/", "dataset/mask_phase3/")

    #rename_images_in_dir("dataset/mask_phase2/")
