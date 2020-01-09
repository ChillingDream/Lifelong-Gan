from time import sleep

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tqdm import trange

from dataset import DataGenerator
from model import Generator
from params import *

G = Generator(input_shape, z_dim)
tl.files.load_and_assign_npz(os.path.join(models_dir, "G_weights_{}.npz".format(model_tag)), G)
tl.files.exists_or_mkdir(os.path.join(sample_dir, "test"))

def test_one_task(test_data, task = ""):
	G.train()
	num_images = len(test_data)
	for i, (image_A, image_B) in zip(trange(0, num_images, batch_size), test_data(batch_size)):
		images = [image_A, image_B]
		for j in range(8):
			z = tf.random.normal(shape=(batch_size, z_dim))
			image = G([image_A, z])
			images.append(image)
		tl.vis.save_images(np.concatenate(images), [len(images) // 2, 2], os.path.join(sample_dir, "test/{}{}.png".format(task, i)))

print("{} tasks in total.".format(len(tasks)))
for i, task in enumerate(tasks):
	print("Task {} ...".format(i + 1))
	test_data = DataGenerator(task, "val")
	test_one_task(test_data, task)
	sleep(1)
print("Tests finishes.")
