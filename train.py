import os
import sys

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

initial_lr = 0.0002
lr_decay = 0.9
beta1 = 0.5
reconst_C = 10
latent_C = 0.5
kl_C = 0.01
image_shape = [image_size, image_size, 3]
tl.files.exists_or_mkdir(save_dir)
tl.files.exists_or_mkdir(models_dir)
tl.files.exists_or_mkdir(log_dir)

G_lr = tf.Variable(initial_lr, dtype=tf.float32, name="G_learning_rate")
D_lr = tf.Variable(initial_lr, dtype=tf.float32, name="D_learning_rate")
E_lr = tf.Variable(initial_lr, dtype=tf.float32, name="E_learning_rate")
G_optimizer = tf.optimizers.Adam(G_lr, beta1)
D_optimizer = tf.optimizers.RMSprop(D_lr, beta1)
E_optimizer = tf.optimizers.Adam(E_lr, beta1)

writer = tf.summary.create_file_writer(log_dir)

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
train_facades = DataGenerator("cityscapes", "train", reverse=True)
print("Training data has been loaded.")

def train_one_task(train_data, use_aux_data = False):
	epoch = 0
	num_images = len(train_data)
	t = trange(0, num_images * epochs, batch_size)

	for step, (image_A, image_B) in zip(t, train_data(batch_size)):
		z = tf.random.normal(shape=(batch_size, z_dim))
		with tf.GradientTape(persistent=True) as tape:
			encoded_z, encoded_mu, encoded_log_sigma = E(image_B)
			vae_img = G([image_A, encoded_z])
			lr_img = G([image_A, z])

			reconst_z, reconst_mu, reconst_log_sigma = E(lr_img)

			P_real = D(image_B)
			P_fake = D(lr_img)
			P_fake_encoded = D(vae_img)

			loss_vae_D = (tl.cost.mean_squared_error(P_real, 0.9, is_mean=True) +
						  tl.cost.mean_squared_error(P_fake_encoded, 0.0, is_mean=True))
			loss_lr_D = (tl.cost.mean_squared_error(P_real, 0.9, is_mean=True) +
						 tl.cost.mean_squared_error(P_fake, 0.0, is_mean=True))
			loss_vae_G = tl.cost.mean_squared_error(P_fake_encoded, 0.9, is_mean=True)
			loss_lr_G = tl.cost.mean_squared_error(P_fake, 0.9, is_mean=True)
			loss_G = loss_vae_G + loss_lr_G
			loss_vae_L1 = tl.cost.absolute_difference_error(
				image_B, vae_img, is_mean=True, axis=[1, 2, 3])
			loss_latent_L1 = tl.cost.absolute_difference_error(z, reconst_z, is_mean=True)
			loss_kl_E = 0.5 * tf.reduce_mean(
				-1 - 2 * encoded_log_sigma + encoded_mu**2 +
				tf.exp(2 * tf.clip_by_value(encoded_log_sigma, -10, 10)))
			loss = loss_vae_D - reconst_C * loss_vae_L1 + loss_lr_D - latent_C * loss_latent_L1 - kl_C * loss_kl_E
			nloss = -loss

			loss_D = loss_vae_D + loss_lr_D
			'''
			loss_G = loss_vae_G + loss_lr_G + reconst_C * loss_vae_L1 + latent_C * loss_latent_L1
			loss_E = loss_vae_G + reconst_C * loss_vae_L1 + latent_C * loss_latent_L1 + kl_C * loss_kl_E
			tot_loss.append([loss_G, loss_GAN_G, loss_D, loss_E])
			'''
		grad = tape.gradient(loss, D.trainable_weights)
		grad, _ = tf.clip_by_global_norm(grad, 10)
		D_optimizer.apply_gradients(zip(grad, D.trainable_weights))

		grad = tape.gradient(nloss, G.trainable_weights)
		grad, _ = tf.clip_by_global_norm(grad, 10)
		G_optimizer.apply_gradients(zip(grad, G.trainable_weights))

		grad = tape.gradient(nloss, E.trainable_weights)
		grad, _ = tf.clip_by_global_norm(grad, 10)
		E_optimizer.apply_gradients(zip(grad, E.trainable_weights))

		del tape
		t.set_description("%f %f %f %f %f" % (loss_G, loss_D, loss_vae_L1, loss_latent_L1, loss_kl_E))
		tf.summary.scalar("loss/loss_GAN_G", loss_G, step)
		tf.summary.scalar("loss/loss_D", loss_D, step)
		tf.summary.scalar("loss/loss_vae_L1", loss_vae_L1, step)
		tf.summary.scalar("loss/loss_latent_L1", loss_latent_L1, step)
		tf.summary.scalar("loss/loss_kl_E", loss_kl_E, step)
		tf.summary.scalar("model/P_real", P_real[0][0], step)
		tf.summary.scalar("model/P_fake", P_fake[0][0], step)
		tf.summary.scalar("model/P_fake_encoded", P_fake_encoded[0][0], step)

		if step % 400 == 0:
			tl.vis.save_images(vae_img.numpy(), [1, batch_size], os.path.join(save_dir, 'vae_g_{}.png'.format(step//400)))
			tl.vis.save_images(lr_img.numpy(), [1, batch_size], os.path.join(save_dir, 'lr_g_{}.png'.format(step//400)))

		if step % num_images == num_images - 1:
			tl.files.save_npz(G.all_weights, os.path.join(models_dir, "G_weights_{}.npz".format(model_tag)))
			tl.files.save_npz(D.all_weights, os.path.join(models_dir, "D_weights_{}.npz".format(model_tag)))
			tl.files.save_npz(E.all_weights, os.path.join(models_dir, "E_weights_{}.npz".format(model_tag)))

			epoch += 1
			new_G_lr = initial_lr * (lr_decay ** epoch)
			new_D_lr = initial_lr * (lr_decay ** epoch)
			G_lr.assign(new_G_lr)
			D_lr.assign(new_D_lr)
			E_lr.assign(new_G_lr)
			tf.summary.scalar("learing_rate/lr_G", G_lr, step)
			tf.summary.scalar("learing_rate/lr_D", D_lr, step)

		writer.flush()

print("Training starts.")
sys.stdout.flush()
with tf.device("/gpu:0"), writer.as_default():
	train_one_task(train_facades, False)
print("Training finishes.")

#G.eval()
#out=G(valid_...).numpy()
