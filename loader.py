import numpy as np
import os
import h5py
import glob
import scipy.misc
from scipy.misc import imread, imresize


def load_images(dataset_dir, option):
	all_imgs = glob.glob(dataset_dir + "/" + option + "/" + "*.jpg")
	img_array_A = []
	img_array_B = []

	for file in all_imgs:
		full_image = imread(file)
		img_B = full_image[:, :full_image.shape[1] // 2, :]
		img_A = full_image[:, full_image.shape[1] // 2:, :]
		img_array_A.append(img_A)
		img_array_B.append(img_B)

	img_array_A = np.asarray(img_array_A) / 255
	img_array_B = np.asarray(img_array_B) / 255
	# print(train_A.shape)
	#print(train_B.shape)

	return (img_array_A, img_array_B)

if __name__ == "__main__":
	train_data = load_images("cityscapes", "train")
	print(np.shape(train_data[1]))