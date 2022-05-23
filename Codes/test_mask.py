import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt

"""simple code for testing how masks were actually saved into directory
print to console which values are actually stored in masks

additionally even show the mask"""

image = load_img("dataset_rokycanska/masks/0000000020.png", target_size=(160, 160), color_mode="grayscale")
img = np.asarray(image)
print(f"unikatni hodnoty reprezentujici tridy masek: {np.unique(img)}")
plt.imshow(img)
plt.show()
