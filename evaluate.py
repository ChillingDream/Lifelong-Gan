import os

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from dataset import DataGenerator
from model import Generator
from params import *

G = Generator((batch_size, 256, 256, 3), z_dim)
tl.files.load_and_assign_npz(os.path.join(models_dir, "G_weights_{}.npz".format(model_tag)), G)

def test_one_task(test_data):
	#G.eval()
	G.train()
	for i, (image_A, image_B) in enumerate(test_data(batch_size)):
		z = tf.random.normal(shape=(batch_size, z_dim))
		image1 = G([image_A, z])
		z = tf.random.normal(shape=(batch_size, z_dim))
		image2 = G([image_A, z])
		tmp = np.concatenate([image_A, image_B, image1, image2])
		print(np.max(image1), np.min(image1), np.max(image2), np.min(image2))
		tl.vis.save_images(np.concatenate([image_A, image_B, image1, image2]), [2, 2], os.path.join(save_dir, "result{}.png".format(i)))

test_data = DataGenerator('facades', 'train', reverse=True)
print("Testing.")
test_one_task(test_data)
