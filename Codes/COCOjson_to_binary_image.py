import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

pic_dims = (1080, 1920)
masks_arr = []
json_mask_paths = "../jsonMasks/"
binary_mask_path = json_mask_paths+'converted_to_binary/'
visualized_masks = json_mask_paths+'converted_informative/'
filenames = []
classes = {}

# load all json files in one dictionary
for i,filename in enumerate(os.listdir(json_mask_paths)):
    f=open(json_mask_paths+filename)
    json_temp = json.load(f)
    if filename == 'classes.json':
        for j in range(len(json_temp)):
            classes[json_temp[j]['id']] = json_temp[j]['name']
        break
    noOfObjects = len(json_temp['instances']) #number of objects in one json anotation
    object_array = [0] * noOfObjects
    for k in range(len(json_temp['instances'])):
        object_in_json = json_temp['instances'][k] #add reference to concrete object in json for easier work
        if object_in_json['type'] == 'polygon':
            object_array[k] = [object_in_json['classId']]
            object_array[k].append(object_in_json['points'])
        else:
            print('Tento program nepodporuje jine tvary nez polygon --> je nutne doprogramovat')
    name = json_temp['metadata']['name'][:-4]
    filenames.append(name)
    masks_arr.append(object_array)

# make images from loaded info
#make directory if not exist
if not os.path.exists(binary_mask_path):
    os.mkdir(binary_mask_path)
if not os.path.exists(visualized_masks):
    os.mkdir(visualized_masks)

for l in range(len(masks_arr)):
    mask_data = masks_arr[l]
    img = np.zeros(pic_dims)
    img_name = filenames[l]
    img_classes = len(mask_data)
    for m in range(img_classes):
        class_color = mask_data[m][0]
        class_points = np.reshape(mask_data[m][1], (-1, 2)).astype(int)
        img = cv2.fillPoly(img, pts=[class_points], color=class_color)
    #print(np.unique(img))
    #plt.imshow(img)
    #plt.show()
    cv2.imwrite(binary_mask_path+img_name+'.png', img)
    plt.imsave(visualized_masks+img_name+'.png', img)
    img = cv2.imread(binary_mask_path+img_name+'.png')
    #print(np.unique(img)) # print vyskytujicich se trid do konzole

