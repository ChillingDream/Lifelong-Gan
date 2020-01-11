import multiprocessing

import numpy as np
import tensorflow as tf
from imageio import imwrite

from loader import load_images
from params import *

dataset_dir = "./data"

class DataGenerator(object):
	def __init__(self, dataset_name, mode):
		self.mode = mode
		if isinstance(dataset_name, list):
			self.images = [[], []]
			for sub_name in dataset_name:
				image_A, image_B = load_images(os.path.join(dataset_dir, sub_name), mode, sub_name in ["facades", "cityscapes"])
				self.images[0].extend(image_A)
				self.images[1].extend(image_B)
			images = list(zip(self.images[0], self.images[1]))
			np.random.shuffle(images)
			self.images[0], self.images[1] = zip(*images)
		else:
			self.images = load_images(os.path.join(dataset_dir, dataset_name), mode, dataset_name in ["facades", "cityscapes"])

	def __len__(self):
		return len(self.images[0])

	def generator(self):
		for imgA, imgB in zip(self.images[0], self.images[1]):
			yield imgA, imgB

	def _map_fn(self, image_A, image_B):
		patch_pnum = image_size // patch_size
		patch_num = patch_pnum * patch_pnum
		sample_origins = list(zip(np.random.choice(range(image_size - patch_size + 1), patch_num),
								  np.random.choice(range(image_size - patch_size + 1), patch_num)))
		aux_A = [tf.slice(image_A, [x, y, 0], [patch_size, patch_size, 3]) for x, y in sample_origins]
		aux_B = [tf.slice(image_B, [x, y, 0], [patch_size, patch_size, 3]) for x, y in sample_origins]
		aux_A = [tf.concat(aux_A[r*patch_pnum:(r+1)*patch_pnum], 1) for r in range(patch_pnum)]
		aux_A = tf.concat(aux_A, 0)
		aux_B = [tf.concat(aux_B[r*patch_pnum:(r+1)*patch_pnum], 1) for r in range(patch_pnum)]
		aux_B = tf.concat(aux_B, 0)
		return image_A, image_B, aux_A, aux_B

	def __call__(self, batch_size, shuffle = True, repeat = True, use_aux = False):
		data = tf.data.Dataset.from_generator(self.generator, output_types=(tf.float32, tf.float32))
		if use_aux:
			data = data.map(self._map_fn, num_parallel_calls=multiprocessing.cpu_count())
		if shuffle == "train":
			data = data.shuffle(100 * batch_size)
		data = data.prefetch(10 * batch_size)
		data = data.batch(batch_size)
		if repeat:
			data = data.repeat()
		return data

if __name__ == '__main__':
	data = DataGenerator('edges2shoes', 'train')
	print("*")
	for i, (a, b, c, d) in zip(range(20), data(1, use_aux=True)):
		print(i)
		print(np.shape(d))
		imwrite("samples/aux_AB{}.png".format(i), np.concatenate([c[0], d[0]], 1))
