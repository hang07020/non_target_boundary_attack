from __future__ import print_function
try:
	raw_input
except:
	raw_input = input

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from keras.datasets import mnist
import pickle
import time
import datetime
import os
from PIL import Image
import json
import cv2

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

def get_diff(sample_1, sample_2):
	sample_1 = sample_1.reshape(3, 224, 224)
	sample_2 = sample_2.reshape(3, 224, 224)
	diff = []
	for i, channel in enumerate(sample_1):
		diff.append(np.linalg.norm((channel - sample_2[i]).astype(np.float32)))
	return np.array(diff)

def preprocess(img):
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x

# 画像の読み込み
image1 = image.load_img('blue_sample/20231003_044605_sea_lion2800.png', target_size=(224, 224))
sample1 = preprocess(image1) #元画像

image2 = image.load_img('original/awkward_moment_seal.png', target_size=(224, 224))
sample2 = preprocess(image2) 

diff = np.mean(get_diff(sample1, sample2))
realdiff = np.sum((sample1/255 - sample2/255 )**2)**0.5

print("Mean Squared Error: {}".format(diff))
print("Real Mean Squared Error: {}".format(realdiff))
