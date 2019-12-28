import multiprocessing
import os

import numpy as np
import tensorflow as tf

from loader import load_images

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
		return image_A, image_B, image_A, image_B

	def __call__(self, batch_size, shuffle = True, repeat = True, use_aux = False):
		data = tf.data.Dataset.from_generator(self.generator, output_types=(tf.float32, tf.float32))
		if use_aux:
			data = data.map(self._map_fn, num_parallel_calls=multiprocessing.cpu_count())
		if self.mode == "train":
			data = data.shuffle(100 * batch_size)
		data = data.prefetch(10 * batch_size)
		data = data.batch(batch_size)
		if repeat:
			data = data.repeat()
		return data
