import os

import tensorflow as tf

from loader import load_images

dataset_dir = "./data"

class DataGenerator(object):
	def __init__(self, dataset_name, mode, reverse):
		self.images = load_images(os.path.join(dataset_dir, dataset_name), mode, reverse)

	def generator(self):
		for imgA, imgB in zip(self.images[0], self.images[1]):
			yield imgA, imgB

	def __call__(self, batch_size, shuffle_size = 0):
		data = tf.data.Dataset.from_generator(self.generator, output_types=(tf.float32, tf.float32))
		data = data.prefetch(buffer_size=20)
		data = data.batch(batch_size)
		return data
