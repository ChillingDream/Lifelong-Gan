import glob
import os

import numpy as np
from scipy.misc import imread, imresize
from tqdm import tqdm

from params import image_shape

def load_images(dataset_path, mode, reverse):
	all_imgs = glob.glob(os.path.join(dataset_path, mode, "*.jpg"))
	img_array_A = []
	img_array_B = []

	for file in tqdm(all_imgs):
		full_image = imread(file)
		if reverse:
			img_B = full_image[:, :full_image.shape[1] // 2, :]
			img_A = full_image[:, full_image.shape[1] // 2:, :]
		else:
			img_A = full_image[:, :full_image.shape[1] // 2, :]
			img_B = full_image[:, full_image.shape[1] // 2:, :]
		img_A = imresize(img_A, image_shape)
		img_B = imresize(img_B, image_shape)
		img_array_A.append(img_A)
		img_array_B.append(img_B)

	img_array_A = (np.asarray(img_array_A).astype(np.float32) / 255 * 2) - 1
	img_array_B = (np.asarray(img_array_B).astype(np.float32) / 255 * 2) - 1

	return img_array_A, img_array_B

if __name__ == "__main__":
	train_data = load_images("cityscapes", "train")
	print(np.shape(train_data[1]))