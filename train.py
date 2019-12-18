import os
import sys
import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tqdm import trange

from dataset import DataGenerator
from model import Generator, Encoder, Discriminator
from params import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)
	tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
	logical_gpus = tf.config.experimental.list_logical_devices('GPU')
	print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

tl.files.exists_or_mkdir(save_dir)
tl.files.exists_or_mkdir(models_dir)
#sample_num=64
#log_dir
#check_point_dir
#result_dir
#lrC = 0.00005  #learning rate
initial_lr = 0.0002
lr_decay = 0.9
beta1 = 0.5
#beta2=0.999
reconst_C = 10
latent_C = 0.5
kl_C = 0.01
image_shape = [image_size, image_size, 3]

G_lr = tf.Variable(initial_lr, dtype=tf.float32, name="G_learning_rate")
D_lr = tf.Variable(initial_lr, dtype=tf.float32, name="D_learning_rate")
E_lr = tf.Variable(initial_lr, dtype=tf.float32, name="E_learning_rate")
G_optimizer = tf.optimizers.Adam(G_lr, beta1)
D_optimizer = tf.optimizers.RMSprop(D_lr, beta1)
E_optimizer = tf.optimizers.Adam(E_lr, beta1)

LOAD = False
G = Generator((batch_size, 256, 256, 3), z_dim)
D = Discriminator((batch_size, 256, 256, 3))
E = Encoder((batch_size, 256, 256, 3), z_dim)
if LOAD:
	tl.files.load_and_assign_npz(os.path.join(models_dir, "G_weights_{}.npz".format(model_tag)), G)
	tl.files.load_and_assign_npz(os.path.join(models_dir, "D_weights_{}.npz".format(model_tag)), D)
	tl.files.load_and_assign_npz(os.path.join(models_dir, "E_weights_{}.npz".format(model_tag)), E)
	print("Model weights has been loaded.")

G.train()
D.train()
E.train()
print("Training data loading.")
train_facades = DataGenerator("facades", "train", reverse=True)
print("Training data has been loaded.")

def train_one_task(train_data, use_aux_data = False):
	start_time = time.time()
	num_images = len(train_data)
	for epoch in range(1, epochs + 1):
		tot_loss = []
		t = trange(0, num_images, batch_size)

		for step, (image_A, image_B) in zip(t, train_data(batch_size)):
			z = tf.random.normal(shape=(batch_size, z_dim))
			with tf.GradientTape(persistent=True) as tape:
				encoded_z, encoded_mu, encoded_log_sigma = E(image_B)
				desired_gen_img = G([image_A, encoded_z])
				LR_desired_img = G([image_A, z])

				reconst_z, reconst_mu, reconst_log_sigma = E(LR_desired_img)

				P_real = D(image_B)
				P_fake = D(LR_desired_img)
				P_fake_encoded = D(desired_gen_img)

				loss_vae_D = (tl.cost.mean_squared_error(P_real, 0.9, is_mean=True) +
							  tl.cost.mean_squared_error(P_fake_encoded, 0.0, is_mean=True))
				loss_lr_D = (tl.cost.mean_squared_error(P_real, 0.9, is_mean=True) +
							 tl.cost.mean_squared_error(P_fake, 0.0, is_mean=True))
				loss_vae_G = tl.cost.mean_squared_error(P_fake_encoded, 0.9, is_mean=True)
				loss_lr_G = tl.cost.mean_squared_error(P_fake, 0.9, is_mean=True)
				loss_GAN_G = loss_vae_G + loss_lr_G
				loss_vae_L1 = tl.cost.absolute_difference_error(
					image_B, desired_gen_img, is_mean=True, axis=[1, 2, 3])
				loss_latent_GE = tl.cost.absolute_difference_error(z, reconst_z, is_mean=True)
				loss_kl_E = 0.5 * tf.reduce_mean(
					-1 - 2 * encoded_log_sigma + encoded_mu**2 +
					tf.exp(2 * tf.clip_by_value(encoded_log_sigma, 0, 10)))

				loss_D = loss_vae_D + loss_lr_D
				loss_G = loss_vae_G + loss_lr_G + reconst_C * loss_vae_L1 + latent_C * loss_latent_GE
				loss_E = loss_vae_G + reconst_C * loss_vae_L1 + latent_C * loss_latent_GE + kl_C * loss_kl_E
				tot_loss.append([loss_G, loss_GAN_G, loss_D, loss_E])

			grad = tape.gradient(loss_D, D.trainable_weights)
			grad, _ = tf.clip_by_global_norm(grad, 10)
			D_optimizer.apply_gradients(zip(grad, D.trainable_weights))

			grad = tape.gradient(loss_G, G.trainable_weights)
			grad, _ = tf.clip_by_global_norm(grad, 10)
			G_optimizer.apply_gradients(zip(grad, G.trainable_weights))

			grad = tape.gradient(loss_E, E.trainable_weights)
			grad, _ = tf.clip_by_global_norm(grad, 10)
			E_optimizer.apply_gradients(zip(grad, E.trainable_weights))
			del tape
			t.set_description("%f %f %f %f %f" % (loss_GAN_G, loss_D, loss_vae_L1, loss_latent_GE, loss_kl_E))

		print("")
		if epoch % 1 == 0:
			tl.files.save_npz(G.all_weights, os.path.join(models_dir, "G_weights_{}.npz".format(model_tag)))
			tl.files.save_npz(D.all_weights, os.path.join(models_dir, "D_weights_{}.npz".format(model_tag)))
			tl.files.save_npz(E.all_weights, os.path.join(models_dir, "E_weights_{}.npz".format(model_tag)))

			tot_loss = np.mean(tot_loss, 0)

			#new_G_lr = tf.clip_by_value(((4 ** tot_loss[1]) * lrC).astype(np.float32), 1e-6, 1e-2)
			#new_D_lr = tf.clip_by_value(((4 ** tot_loss[2]) * lrC).astype(np.float32), 1e-6, 1e-2)
			new_G_lr = initial_lr * (lr_decay ** epoch)
			new_D_lr = initial_lr * (lr_decay ** epoch)
			G_lr.assign(new_G_lr)
			D_lr.assign(new_D_lr)
			E_lr.assign(new_G_lr)
			print("time_used=%f" % (time.time() - start_time))
			print("Epoch %d finished! loss_G=%f loss_D=%f loss_E=%f" % (epoch, tot_loss[0], tot_loss[2], tot_loss[3]))
			print("New learning rates are %f %f" % (new_G_lr, new_D_lr))
			print("")
			sys.stdout.flush()

			tl.vis.save_images(desired_gen_img.numpy(), [1, batch_size],
							   os.path.join(save_dir,
											'vae_g_{}.png'.format(epoch)))
			tl.vis.save_images(LR_desired_img.numpy(), [1, batch_size],
							   os.path.join(save_dir,
											'lr_g_{}.png'.format(epoch)))

print("Training starts.")
sys.stdout.flush()
with tf.device("/gpu:0"):
	train_one_task(train_facades, False)
print("Training finishes.")

#G.eval()
#out=G(valid_...).numpy()
