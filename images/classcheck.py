import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

def preprocess(sample_path):
	img = image.load_img(sample_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x

def class_check():
	classifier = ResNet50(weights='imagenet')
	"""sample = preprocess('original/awkward_moment_seal.png')"""
	sample = preprocess('20230920_121746/20230920_134727_sea_lion.png')
	target_class = np.argmax(classifier.predict(sample))
	label = decode_predictions(classifier.predict(sample), top=1)[0][0][1]
	print(target_class)
	print(label)


if __name__ == "__main__":
	class_check()