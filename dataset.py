import os

import tensorflow as tf

from loader import load_images

dataset_dir = "./data"

class DataGenerator(object):
	def __init__(self, dataset_name, mode, reverse):
		self.mode = mode
		self.images = load_images(os.path.join(dataset_dir, dataset_name), mode, reverse)

	def __len__(self):
		return len(self.images[0])

	def generator(self):
		for imgA, imgB in zip(self.images[0], self.images[1]):
			yield imgA, imgB

	def __call__(self, batch_size, shuffle = True, repeat = True):
		data = tf.data.Dataset.from_generator(self.generator, output_types=(tf.float32, tf.float32))
		if self.mode == "train":
			data = data.shuffle(100 * batch_size)
		data = data.prefetch(10 * batch_size)
		data = data.batch(batch_size)
		if repeat:
			data = data.repeat()
		return data
