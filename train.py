import os
import sys
import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from dataset import DataGenerator
from loader import anti_std
from model import Generator, Encoder, Discriminator

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)
	tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
	logical_gpus = tf.config.experimental.list_logical_devices('GPU')
	print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

epochs = 20
batch_size = 4
image_size = 256
Z_dim = 8
save_dir = 'samples/facades'
models_dir = 'nets/facades'
model_tag = 2
tl.files.exists_or_mkdir(save_dir)
tl.files.exists_or_mkdir(models_dir)
#sample_num=64
#log_dir
#check_point_dir
#result_dir
lrC = 0.0002  #learning rate
beta1 = 0.5
#beta2=0.999
lr_updateC = 0.01
reconst_C = 10
latent_C = 0.5
kl_C = 0.01
image_dims = [256, 256, 3]
z_shape0 = (batch_size, 1, 1, Z_dim)
z_shape1 = (1, image_size, image_size, 1)

G_lr = tf.Variable(lrC, dtype=tf.float32, name="G_learning_rate")
D_lr = tf.Variable(lrC, dtype=tf.float32, name="D_learning_rate")
E_lr = tf.Variable(lrC, dtype=tf.float32, name="E_learning_rate")
G_optimizer = tf.optimizers.Adam(G_lr, beta1)
D_optimizer = tf.optimizers.Adam(D_lr, beta1)
E_optimizer = tf.optimizers.Adam(E_lr, beta1)

LOAD = True
G = Generator((batch_size, 256, 256, 3 + Z_dim))
D = Discriminator((batch_size, 256, 256, 3))
E = Encoder((batch_size, 256, 256, 3), Z_dim)
if LOAD:
	tl.files.load_and_assign_npz(os.path.join(models_dir, "G_weights_{}.npz".format(model_tag)), G)
	tl.files.load_and_assign_npz(os.path.join(models_dir, "D_weights_{}.npz".format(model_tag)), D)
	tl.files.load_and_assign_npz(os.path.join(models_dir, "E_weights_{}.npz".format(model_tag)), E)
	print("Model weights has been loaded.")

G.train()
D.train()
E.train()
print("Training data loading.")
train_facades = DataGenerator("facades", "train", False)
print("Training data has been loaded.")

def train_one_task(train_data, use_aux_data = False):
	start_time = time.time()
	for epoch in range(1, epochs + 1):
		tot_loss = []
		processed = 0
		for image_A, image_B in train_data(batch_size):
			z = tf.random.normal(shape=(batch_size, Z_dim))

			with tf.GradientTape(persistent=True) as tape:
				encoded_true_img_z, encoded_mu, encoded_log_sigma = E(image_B)

				reshaped_z = tf.reshape(z, z_shape0)
				tiled_z = tf.tile(reshaped_z, z_shape1)
				encoded_true_img_z = tf.reshape(encoded_true_img_z, z_shape0)
				encoded_true_img_z = tf.tile(encoded_true_img_z, z_shape1)
				encoded_GI = tf.concat([image_A, encoded_true_img_z], axis=3)
				GI = tf.concat([image_A, tiled_z], axis=3)
				desired_gen_img = G(encoded_GI)
				LR_desired_img = G(GI)

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
			processed += batch_size
			#print("%f %f %f %f %f" % (loss_GAN_G, loss_D, loss_vae_L1, loss_latent_GE, loss_kl_E))
			#sys.stdout.flush()
			if processed > 100:
			#	print("#", end="")
				print("%f %f %f %f %f" % (loss_GAN_G, loss_D, loss_vae_L1, loss_latent_GE, loss_kl_E))
				sys.stdout.flush()
				processed -= 100

		print("")
		if epoch % 1 == 0:
			tl.files.save_npz(G.all_weights, os.path.join(models_dir, "G_weights_{}.npz".format(model_tag)))
			tl.files.save_npz(D.all_weights, os.path.join(models_dir, "D_weights_{}.npz".format(model_tag)))
			tl.files.save_npz(E.all_weights, os.path.join(models_dir, "E_weights_{}.npz".format(model_tag)))

			tot_loss = np.mean(tot_loss, 0)

			new_G_lr = tf.clip_by_value((lr_updateC * (10 ** tot_loss[1]) * lrC).astype(np.float32), 1e-6, 1e-2)
			new_D_lr = tf.clip_by_value((lr_updateC * (10 ** tot_loss[2]) * lrC).astype(np.float32), 1e-6, 1e-2)
			G_lr.assign(new_G_lr)
			D_lr.assign(new_D_lr)
			E_lr.assign(new_G_lr)
			print("time_used=%f" % (time.time() - start_time))
			print("Epoch %d finished! loss_G=%f loss_D=%f loss_E=%f" % (epoch, tot_loss[0], tot_loss[2], tot_loss[3]))
			print("New learning rates are %f %f" % (new_G_lr, new_D_lr))
			print("")
			sys.stdout.flush()

			tl.vis.save_images(anti_std(desired_gen_img.numpy()), [1, batch_size],
							   os.path.join(save_dir,
											'vae_g_{}.png'.format(epoch)))
			tl.vis.save_images(anti_std(LR_desired_img.numpy()), [1, batch_size],
							   os.path.join(save_dir,
											'lr_g_{}.png'.format(epoch)))

print("Training starts.")
sys.stdout.flush()
train_one_task(train_facades, False)
print("Training finishes.")

#G.eval()
#out=G(valid_...).numpy()
