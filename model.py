import os

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.models import Model

from params import *

lrelu = lambda x: tl.act.lrelu(x, 0.2)


def residual(R, n_f, f_s):
	w_init = tl.initializers.truncated_normal(stddev=0.01)
	R_tmp = R
	R = BatchNorm2d(act=tf.nn.relu)(Conv2d(n_f, (f_s, f_s), (1, 1), W_init=w_init)(R))
	R = BatchNorm2d(act=None)(Conv2d(n_f, (f_s, f_s), (1, 1), W_init=w_init)(R))
	R_tmp = Conv2d(n_f, (1, 1), (1, 1))(R_tmp)
	return Elementwise(tf.add, act=tf.nn.relu)([R_tmp, R])


def Discriminator(input_shape, prefix = ""):
	I = Input(input_shape)
	D = Conv2d(
		64, (4, 4), (2, 2), padding='SAME', act=lrelu, b_init=None, name=prefix+'D_conv_1')(I)
	D = InstanceNorm2d(act=lrelu)(Conv2d(
		128, (4, 4), (2, 2), padding='SAME', b_init=None, name=prefix+'D_conv_2')(D))
	D = InstanceNorm2d(act=lrelu)(Conv2d(
		256, (4, 4), (2, 2), padding='SAME', b_init=None, name=prefix+'D_conv_3')(D))
	D = InstanceNorm2d(act=lrelu)(Conv2d(
		512, (4, 4), (2, 2), padding='SAME', b_init=None, name=prefix+'D_conv_4')(D))
	D = InstanceNorm2d(act=lrelu)(Conv2d(
		512, (4, 4), (2, 2), padding='SAME', b_init=None, name=prefix+'D_conv_5')(D))
	D = InstanceNorm2d(act=lrelu)(Conv2d(
		512, (4, 4), (2, 2), padding='SAME', b_init=None, name=prefix+'D_conv_6')(D))
	D = Conv2d(1, (4, 4), (1, 1), name=prefix+'D_conv_7')(D)
	D = GlobalMeanPool2d()(D)
	D_net = Model(inputs=I, outputs=D, name=prefix+'Discriminator')
	return D_net


def Generator(input_shape, z_dim, prefix = ""):
	w_init = tl.initializers.truncated_normal(stddev=0.01)
	I = Input(input_shape)
	Z = Input([input_shape[0], z_dim])
	z = Reshape((input_shape[0], 1, 1, -1))(Z)
	z = Tile([1, input_shape[1], input_shape[2], 1])(z)

	conv_layers = []
	G = Concat(concat_dim=-1)([I, z])
	filters = [64, 128, 256, 512, 512, 512, 512]
	if image_size == 256:
		filters.append(512)
	G = Conv2d(
		filters[0], (4, 4), (2, 2), act=lrelu, W_init=w_init, b_init=None, name=prefix+'G_conv_1')(G)
	conv_layers.append(G)
	for i, n_f in enumerate(filters[1:]):
		G = BatchNorm2d(act=lrelu)(Conv2d(
			n_f, (4, 4), (2, 2), W_init=w_init, b_init=None, name=prefix+'G_conv_{}'.format(i + 2))(G))
		conv_layers.append(G)

	filters.pop()
	filters.reverse()
	conv_layers.pop()
	for i, n_f in enumerate(filters):
		G = BatchNorm2d(act=tf.nn.relu)(DeConv2d(
			n_f, (4, 4), (2, 2), W_init=w_init, b_init=None, name=prefix+'G_deconv_{}'.format(len(filters)+1-i))(G))
		G = Concat(concat_dim=-1)([G, conv_layers.pop()])
	G = DeConv2d(3, (4, 4), (2, 2), act=tf.nn.tanh, W_init=w_init, b_init=None, name=prefix+'G_deconv_1')(G)
	G_net = Model(inputs=[I, Z], outputs=G, name=prefix+'Generator')
	return G_net


def Encoder(input_shape, z_dim, prefix=""):
	I = Input(input_shape)
	E = Conv2d(64, (4, 4), (2, 2), act=lrelu, name=prefix+'E_conv_1')(I)
	E = MeanPool2d((2, 2), (2, 2), 'SAME')(residual(E, 128, 3))
	E = MeanPool2d((2, 2), (2, 2), 'SAME')(residual(E, 256, 3))
	E = MeanPool2d((2, 2), (2, 2), 'SAME')(residual(E, 512, 3))
	E = MeanPool2d((2, 2), (2, 2), 'SAME')(residual(E, 512, 3))
	E = MeanPool2d((2, 2), (2, 2), 'SAME')(residual(E, 512, 3))
	E = Flatten()(MeanPool2d((8, 8), (8, 8), 'SAME')(E))
	mu = Dense(z_dim)(E)
	log_sigma = Dense(z_dim)(E)
	z = Elementwise(tf.add)(
		[mu, Lambda(lambda x:tf.random.normal(shape=[z_dim]) * tf.exp(x))(log_sigma)])
	E_net = Model(inputs=I, outputs=[z, mu, log_sigma], name=prefix+'Encoder')
	return E_net

class BicycleGAN(object):
	count = 0

	def __init__(self, LOAD = False):
		BicycleGAN.count += 1
		self.G = Generator(input_shape, z_dim, "model_{}/".format(self.count))
		self.D = Discriminator(input_shape, "model_{}/".format(self.count))
		self.E = Encoder(input_shape, z_dim, "model_{}/".format(self.count))
		if LOAD:
			self.load(model_tag)
		self.G.train()
		self.D.train()
		self.E.train()

	def load(self, model_tag):
		tl.files.load_and_assign_npz(os.path.join(models_dir, "G_weights_{}.npz".format(model_tag)), self.G)
		tl.files.load_and_assign_npz(os.path.join(models_dir, "D_weights_{}.npz".format(model_tag)), self.D)
		tl.files.load_and_assign_npz(os.path.join(models_dir, "E_weights_{}.npz".format(model_tag)), self.E)
		print("Model weights has been loaded.")

	def save(self, model_tag):
		tl.files.save_npz(self.G.all_weights, os.path.join(models_dir, "G_weights_{}.npz".format(model_tag)))
		tl.files.save_npz(self.D.all_weights, os.path.join(models_dir, "D_weights_{}.npz".format(model_tag)))
		tl.files.save_npz(self.E.all_weights, os.path.join(models_dir, "E_weights_{}.npz".format(model_tag)))

	def calc_loss(self, image_A, image_B, z):
		encoded_z, encoded_mu, encoded_log_sigma = self.E(image_B)
		vae_img = self.G([image_A, encoded_z])
		lr_img = self.G([image_A, z])
		self.vae_img = vae_img
		self.lr_img = lr_img

		reconst_z, reconst_mu, reconst_log_sigma = self.E(lr_img)

		P_real = self.D(image_B)
		P_fake = self.D(lr_img)
		P_fake_encoded = self.D(vae_img)
		self.P_real = P_real
		self.P_fake = P_fake
		self.P_fake_encoded = P_fake_encoded

		loss_vae_D = (tl.cost.mean_squared_error(P_real, 0.9, is_mean=True) +
					  tl.cost.mean_squared_error(P_fake_encoded, 0.0, is_mean=True))
		loss_lr_D = (tl.cost.mean_squared_error(P_real, 0.9, is_mean=True) +
					 tl.cost.mean_squared_error(P_fake, 0.0, is_mean=True))
		loss_vae_G = tl.cost.mean_squared_error(P_fake_encoded, 0.9, is_mean=True)
		loss_lr_G = tl.cost.mean_squared_error(P_fake, 0.9, is_mean=True)
		self.loss_G = loss_vae_G + loss_lr_G
		self.loss_D = loss_vae_D + loss_lr_D
		self.loss_vae_L1 = tl.cost.absolute_difference_error(
			image_B, vae_img, is_mean=True, axis=[1, 2, 3])
		self.loss_latent_L1 = tl.cost.absolute_difference_error(z, reconst_z, is_mean=True)
		self.loss_kl_E = 0.5 * tf.reduce_mean(
			-1 - 2 * encoded_log_sigma + encoded_mu**2 +
			tf.exp(2 * tf.clip_by_value(encoded_log_sigma, -10, 10)))
		loss = self.loss_D - reconst_C * self.loss_vae_L1 - latent_C * self.loss_latent_L1 - kl_C * self.loss_kl_E
		return loss

if __name__ == '__main__':
	x = BicycleGAN()
	print(x.G)
	print(x.D)
	print(x.E)
	y = BicycleGAN()
	print(y.G)
	print(y.D)
	print(y.E)

