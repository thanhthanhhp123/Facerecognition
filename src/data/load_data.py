
"""
Created on Sat Mar  4 09:54:25 2023

@author: thanhdeptrai
"""
from PIL import Image
import os
import numpy as np
def load_data():
    X = []
    y = []
    data_dir = '/Users/tranthanh/Downloads/FaceRecognition/FaceDatasets'
    subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    for label, folder in enumerate(subfolders):
        for file_name in os.listdir(folder):
            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                image = Image.open(os.path.join(folder, file_name))
                image_array = np.array(image)
                X.append(image_array)
                y.append(label)
    X = np.array(X)
    y = np.array(y)
    classes = os.listdir(data_dir)
    return X, y, classes