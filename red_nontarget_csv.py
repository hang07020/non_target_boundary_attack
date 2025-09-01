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
import pandas as pd
import random

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions


def orthogonal_perturbation(delta, prev_sample, target_sample):
	prev_sample = prev_sample.reshape(224, 224, 3)
	# Generate perturbation
	perturb = np.random.randn(224, 224, 3)
	perturb /= get_diff(perturb, np.zeros_like(perturb))
	perturb *= delta * np.mean(get_diff(target_sample, prev_sample))
	# Project perturbation onto sphere around target
	diff = (target_sample - prev_sample).astype(np.float32)
	diff /= get_diff(target_sample, prev_sample)
	diff = diff.reshape(3, 224, 224)
	perturb = perturb.reshape(3, 224, 224)
	for i, channel in enumerate(diff):
		perturb[i] -= np.dot(perturb[i], channel) * channel
	# Check overflow and underflow
	mean = [103.939, 116.779, 123.68]
	perturb = perturb.reshape(224, 224, 3)
	overflow = (prev_sample + perturb) - np.concatenate((np.ones((224, 224, 1)) * (255. - mean[0]), np.ones((224, 224, 1)) * (255. - mean[1]), np.ones((224, 224, 1)) * (255. - mean[2])), axis=2)
	overflow = overflow.reshape(224, 224, 3)
	perturb -= overflow * (overflow > 0)
	underflow = np.concatenate((np.ones((224, 224, 1)) * (0. - mean[0]), np.ones((224, 224, 1)) * (0. - mean[1]), np.ones((224, 224, 1)) * (0. - mean[2])), axis=2) - (prev_sample + perturb)
	underflow = underflow.reshape(224, 224, 3)
	perturb += underflow * (underflow > 0)
	return perturb

def forward_perturbation(epsilon, prev_sample, target_sample):
	perturb = (target_sample - prev_sample).astype(np.float32)
	perturb /= get_diff(target_sample, prev_sample)
	perturb *= epsilon
	return perturb

def get_converted_prediction(sample, classifier):
	sample = sample.reshape(224, 224, 3)
	mean = [103.939, 116.779, 123.68]
	sample[..., 0] += mean[0]
	sample[..., 1] += mean[1]
	sample[..., 2] += mean[2]
	sample = sample[..., ::-1].astype(np.uint8)
	sample = sample.astype(np.float32).reshape(1, 224, 224, 3)
	sample = sample[..., ::-1]
	mean = [103.939, 116.779, 123.68]
	sample[..., 0] -= mean[0]
	sample[..., 1] -= mean[1]
	sample[..., 2] -= mean[2]
	label = decode_predictions(classifier.predict(sample), top=1)[0][0][1]
	return label

def draw(sample, classifier, folder):
	label = get_converted_prediction(np.copy(sample), classifier)
	sample = sample.reshape(224, 224, 3)
	# Reverse preprocessing, see https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
	mean = [103.939, 116.779, 123.68]
	sample[..., 0] += mean[0]
	sample[..., 1] += mean[1]
	sample[..., 2] += mean[2]
	sample = sample[..., ::-1].astype(np.uint8)
	# Convert array to image and save
	sample = Image.fromarray(sample)
	id_no = time.strftime('%Y%m%d_%H%M%S', datetime.datetime.now().timetuple())
	# Save with predicted label for image (may not be adversarial due to uint8 conversion)
	sample.save(os.path.join("images", folder, "{}_{}.png".format(id_no, label)))

def preprocess(img):
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x

def get_diff(sample_1, sample_2):
	sample_1 = sample_1.reshape(3, 224, 224)
	sample_2 = sample_2.reshape(3, 224, 224)
	diff = []
	for i, channel in enumerate(sample_1):
		diff.append(np.linalg.norm((channel - sample_2[i]).astype(np.float32)))
	return np.array(diff)

def get_sc_img(initial_sample):#初期画像の単色画像を生成する

	#RGBの数値
	red = 255
	green = 0
	blue = 0

	height, width= 224,224
	mean_sample = Image.new('RGB', (width, height), (red, green, blue))

	mean_sample.save('single_color_img.png')
	return mean_sample

def boundary_attack():
	classifier = ResNet50(weights='imagenet')
	input_image = image.load_img('images/original/awkward_moment_seal.png', target_size=(224, 224))
	initial_sample = preprocess(input_image) #iniital_image
	target_sample = preprocess(get_sc_img(input_image)) #target_image
	
	results = pd.DataFrame(columns=['Step', 'MSE', 'RMSE'])

	folder = time.strftime('%Y%m%d_%H%M%S', datetime.datetime.now().timetuple())
	os.mkdir(os.path.join("images", folder))
	draw(np.copy(initial_sample), classifier, folder)
	attack_class = np.argmax(classifier.predict(initial_sample))
	target_class = np.argmax(classifier.predict(target_sample))
	print(attack_class)

	adversarial_sample = initial_sample
	n_steps = 0
	n_calls = 0
	epsilon = 1.
	delta = 0.1

	# Move first step to the boundary 
	while True:
		trial_sample = adversarial_sample + forward_perturbation(epsilon * get_diff(adversarial_sample, target_sample), adversarial_sample, target_sample)
		prediction = classifier.predict(trial_sample.reshape(1, 224, 224, 3))
		print(np.argmax(prediction))
		n_calls += 1
		if np.argmax(prediction) == attack_class:
			epsilon /= 0.9
			trial_sample = adversarial_sample + forward_perturbation(epsilon * get_diff(adversarial_sample, target_sample), adversarial_sample, target_sample)
			adversarial_sample = trial_sample			
			draw(np.copy(adversarial_sample), classifier, folder)
			break
		else:
			epsilon *= 0.9
	
	target_class = np.argmax(classifier.predict(adversarial_sample))
	print(target_class)
	print(epsilon)
	diff = np.mean(get_diff(adversarial_sample, initial_sample))
	realdiff = np.sum((adversarial_sample/255 - initial_sample/255 )**2)**0.5
	print("Mean Squared Error: {}".format(diff))
	print("Real Mean Squared Error: {}".format(realdiff))

	while n_steps < 2000:
		print("Step #{}...".format(n_steps))
		print("\tDelta step...")
		d_step = 0
		while True:
			d_step += 1
			print("\t#{}".format(d_step))
			trial_samples = []
			for i in np.arange(10):
				trial_sample = adversarial_sample + orthogonal_perturbation(delta, adversarial_sample, initial_sample)
				trial_samples.append(trial_sample)
			predictions = classifier.predict(np.array(trial_samples).reshape(-1, 224, 224, 3))
			n_calls += 10
			predictions = np.argmax(predictions, axis=1)
			d_score = np.mean(predictions == target_class) 
			if d_score > 0.0:
				if d_score < 0.3:
					delta *= 0.9
				elif d_score > 0.7:
					delta /= 0.9
				adversarial_sample = np.array(trial_samples)[np.where(predictions == target_class)[0][0]]
				break
			else:
				delta *= 0.9
		print("\tEpsilon step...")
		e_step = 0
		while True:
			e_step += 1
			print("\t#{}".format(e_step))
			trial_sample = adversarial_sample + forward_perturbation(epsilon * get_diff(adversarial_sample, initial_sample), adversarial_sample, initial_sample)
			prediction = classifier.predict(trial_sample.reshape(1, 224, 224, 3))
			n_calls += 1
			if np.argmax(prediction) == target_class:
				adversarial_sample = trial_sample
				epsilon /= 0.5
				break
			elif e_step > 500:
					break
			else:
				epsilon *= 0.5
		n_steps += 1
		chkpts = [1, 50]
		if (n_steps in chkpts) or (n_steps % 50 == 0):
			mse = np.mean(get_diff(adversarial_sample, initial_sample))
			rmse = np.sum((adversarial_sample/255 - initial_sample/255 )**2)**0.5
			step_results = pd.DataFrame({'Step': [n_steps], 'MSE': [mse], 'RMSE': [rmse]})
			results = pd.concat([results, step_results], ignore_index=True)
			print("{} steps".format(n_steps))
			draw(np.copy(adversarial_sample), classifier, folder)
		diff = np.mean(get_diff(adversarial_sample, initial_sample))
		realdiff = np.sum((adversarial_sample/255 - initial_sample/255 )**2)**0.5
		if diff <= 1e-3 or e_step > 500:
			print("{} steps".format(n_steps))
			print("Mean Squared Error: {}".format(diff))
			draw(np.copy(adversarial_sample), classifier, folder)
			break
		print("Mean Squared Error: {}".format(diff))
		print("Real Mean Squared Error: {}".format(realdiff))
		print("Calls: {}".format(n_calls))
		print("Attack Class: {}".format(attack_class))
		print("Target Class: {}".format(target_class))
		print("Adversarial Class: {}".format(np.argmax(prediction)))

	csv_file_path = os.path.join("images", folder)
	csv_file_path = os.path.join(csv_file_path, 'results.csv')
	results.to_csv(csv_file_path, index=False)

if __name__ == "__main__":

	boundary_attack()
