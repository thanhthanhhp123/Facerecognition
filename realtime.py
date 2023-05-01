import numpy as np
import pickle
import cv2


model = pickle.load(open('src/models/SVM_model.pkl', 'rb'))
classes = ['dong', 'khai', 'quan', 'thanh']