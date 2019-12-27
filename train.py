import os
import sys

import tensorflow as tf
import tensorlayer as tl
from tqdm import trange

from dataset import DataGenerator
from model import BicycleGAN
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
tl.files.exists_or_mkdir(log_dir)

writer = tf.summary.create_file_writer(log_dir)

LOAD = False
bicycleGAN = BicycleGAN(LOAD)
preGAN = BicycleGAN(LOAD)

def train_one_task(train_data, use_aux_data = False):
	G_lr = tf.Variable(initial_lr, dtype=tf.float32, name="G_learning_rate")
	D_lr = tf.Variable(initial_lr, dtype=tf.float32, name="D_learning_rate")
	E_lr = tf.Variable(initial_lr, dtype=tf.float32, name="E_learning_rate")
	G_optimizer = tf.optimizers.Adam(G_lr, beta1)
	D_optimizer = tf.optimizers.RMSprop(D_lr, beta1)
	E_optimizer = tf.optimizers.Adam(E_lr, beta1)

	epoch = 0
	num_images = len(train_data)
	t = trange(0, num_images * epochs, batch_size)

	for step, _ in zip(t, train_data(batch_size, use_aux_data)):
		if use_aux_data:
			image_A, image_B, aux_A, aux_B = _
		else:
			image_A, image_B = _
		z = tf.random.normal(shape=(batch_size, z_dim))
		with tf.GradientTape(persistent=True) as tape:
			loss = bicycleGAN.calc_loss(image_A, image_B, z)
			nloss = -loss
		grad = tape.gradient(loss, bicycleGAN.D.trainable_weights)
		grad, norm_D = tf.clip_by_global_norm(grad, 10)
		tf.summary.scalar("gradients_norm/norm_D", norm_D, step)
		D_optimizer.apply_gradients(zip(grad, bicycleGAN.D.trainable_weights))

		grad = tape.gradient(nloss, bicycleGAN.G.trainable_weights)
		grad, norm_G = tf.clip_by_global_norm(grad, 10)
		tf.summary.scalar("gradients_norm/norm_G", norm_G, step)
		G_optimizer.apply_gradients(zip(grad, bicycleGAN.G.trainable_weights))

		grad = tape.gradient(nloss, bicycleGAN.E.trainable_weights)
		grad, norm_E = tf.clip_by_global_norm(grad, 10)
		tf.summary.scalar("gradients_norm/norm_E", norm_E, step)
		E_optimizer.apply_gradients(zip(grad, bicycleGAN.E.trainable_weights))

		del tape
		#t.set_description("%f %f %f %f %f" % (loss_G, loss_D, loss_vae_L1, loss_latent_L1, loss_kl_E))
		tf.summary.scalar("loss/loss_GAN_G", bicycleGAN.loss_G, step)
		tf.summary.scalar("loss/loss_D", bicycleGAN.loss_D, step)
		tf.summary.scalar("loss/loss_vae_L1", bicycleGAN.loss_vae_L1, step)
		tf.summary.scalar("loss/loss_latent_L1", bicycleGAN.loss_latent_L1, step)
		tf.summary.scalar("loss/loss_kl_E", bicycleGAN.loss_kl_E, step)
		tf.summary.scalar("model/P_real", bicycleGAN.P_real[0][0], step)
		tf.summary.scalar("model/P_fake", bicycleGAN.P_fake[0][0], step)
		tf.summary.scalar("model/P_fake_encoded", bicycleGAN.P_fake_encoded[0][0], step)

		if step % 400 == 0:
			tl.vis.save_images(bicycleGAN.vae_img.numpy(), [1, batch_size], os.path.join(save_dir, 'vae_g_{}.png'.format(step//400)))
			tl.vis.save_images(bicycleGAN.lr_img.numpy(), [1, batch_size], os.path.join(save_dir, 'lr_g_{}.png'.format(step//400)))

		if step % num_images == num_images - 1:
			bicycleGAN.save(model_tag)

			epoch += 1
			new_G_lr = initial_lr * (lr_decay_G ** epoch)
			new_D_lr = initial_lr * (lr_decay_D ** epoch)
			G_lr.assign(new_G_lr)
			D_lr.assign(new_D_lr)
			E_lr.assign(new_G_lr)
			tf.summary.scalar("learing_rate/lr_G", G_lr, step)
			tf.summary.scalar("learing_rate/lr_D", D_lr, step)

		writer.flush()

print("Training data loading.")
train_facades = DataGenerator("facades", "train", reverse=True)
print("Training data has been loaded.")
print("Training starts.")
sys.stdout.flush()
with tf.device("/gpu:0"), writer.as_default():
	train_one_task(train_facades, False)
print("Training finishes.")

