import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.models import Model

lrelu = lambda x: tl.act.lrelu(x, 0.2)


def residual(R, n_f, f_s):
	w_init = tl.initializers.truncated_normal(stddev=0.01)
	R_tmp = R
	R = BatchNorm2d(act=tf.nn.relu)(Conv2d(n_f, (f_s, f_s), (1, 1), W_init=w_init)(R))
	R = BatchNorm2d(act=None)(Conv2d(n_f, (f_s, f_s), (1, 1), W_init=w_init)(R))
	R_tmp = Conv2d(n_f, (1, 1), (1, 1))(R_tmp)
	return Elementwise(tf.add, act=tf.nn.relu)([R_tmp, R])


def Discriminator(input_shape):
	I = Input(input_shape)
	D = Conv2d(
		64, (4, 4), (2, 2), padding='SAME', act=lrelu, b_init=None, name='D_conv_1')(I)
	D = InstanceNorm2d(act=lrelu)(Conv2d(
		128, (4, 4), (2, 2), padding='SAME', b_init=None, name='D_conv_2')(D))
	D = InstanceNorm2d(act=lrelu)(Conv2d(
		256, (4, 4), (2, 2), padding='SAME', b_init=None, name='D_conv_3')(D))
	D = InstanceNorm2d(act=lrelu)(Conv2d(
		512, (4, 4), (2, 2), padding='SAME', b_init=None, name='D_conv_4')(D))
	D = InstanceNorm2d(act=lrelu)(Conv2d(
		512, (4, 4), (2, 2), padding='SAME', b_init=None, name='D_conv_5')(D))
	D = InstanceNorm2d(act=lrelu)(Conv2d(
		512, (4, 4), (2, 2), padding='SAME', b_init=None, name='D_conv_6')(D))
	D = Conv2d(1, (4, 4), (1, 1), name='D_conv_7')(D)
	D = GlobalMeanPool2d()(D)
	D_net = Model(inputs=I, outputs=D, name='Discriminator')
	return D_net


def Generator(input_shape, z_dim):
	w_init = tl.initializers.truncated_normal(stddev=0.01)
	I = Input(input_shape)
	Z = Input([input_shape[0], z_dim])
	z = Reshape((input_shape[0], 1, 1, -1))(Z)
	z = Tile([1, input_shape[1], input_shape[2], 1])(z)

	conv_layers = []
	G = Concat(concat_dim=-1)([I, z])
	G = Conv2d(
		64, (4, 4), (2, 2), padding='SAME', act=lrelu, W_init=w_init, b_init=None, name='G_conv_1')(G)
	conv_layers.append(G)
	G = BatchNorm2d(act=lrelu)(Conv2d(
		128, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None, name='G_conv_2')(G))
	conv_layers.append(G)
	G = BatchNorm2d(act=lrelu)(Conv2d(
		256, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None, name='G_conv_3')(G))
	conv_layers.append(G)
	G = BatchNorm2d(act=lrelu)(Conv2d(
		512, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None, name='G_conv_4')(G))
	conv_layers.append(G)
	G = BatchNorm2d(act=lrelu)(Conv2d(
		512, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None, name='G_conv_5')(G))
	conv_layers.append(G)
	G = BatchNorm2d(act=lrelu)(Conv2d(
		512, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None, name='G_conv_6')(G))
	conv_layers.append(G)
	G = BatchNorm2d(act=lrelu)(Conv2d(
		512, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None, name='G_conv_7')(G))
	conv_layers.append(G)
	G = BatchNorm2d(act=lrelu)(Conv2d(
		512, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None, name='G_conv_8')(G))

	G = BatchNorm2d(act=tf.nn.relu)(DeConv2d(
		512, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None, name='G_deconv_8')(G))
	G = Concat(concat_dim=-1)([G, conv_layers.pop()])
	G = BatchNorm2d(act=tf.nn.relu)(DeConv2d(
		512, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None, name='G_deconv_7')(G))
	G = Concat(concat_dim=-1)([G, conv_layers.pop()])
	G = BatchNorm2d(act=tf.nn.relu)(DeConv2d(
		512, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None, name='G_deconv_6')(G))
	G = Concat(concat_dim=-1)([G, conv_layers.pop()])
	G = BatchNorm2d(act=tf.nn.relu)(DeConv2d(
		512, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None, name='G_deconv_5')(G))
	G = Concat(concat_dim=-1)([G, conv_layers.pop()])
	G = BatchNorm2d(act=tf.nn.relu)(DeConv2d(
		256, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None, name='G_deconv_4')(G))
	G = Concat(concat_dim=-1)([G, conv_layers.pop()])
	G = BatchNorm2d(act=tf.nn.relu)(DeConv2d(
		128, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None, name='G_deconv_3')(G))
	G = Concat(concat_dim=-1)([G, conv_layers.pop()])
	G = BatchNorm2d(act=tf.nn.relu)(DeConv2d(
		64, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None, name='G_deconv_2')(G))
	G = Concat(concat_dim=-1)([G, conv_layers.pop()])
	G = DeConv2d(3, (4, 4), (2, 2), padding='SAME', act=tf.nn.tanh, W_init=w_init, b_init=None, name='G_deconv_1')(G)
	G_net = Model(inputs=[I, Z], outputs=G, name='Generator')
	return G_net


def Encoder(input_shape, z_dim):
	I = Input(input_shape)
	E = Conv2d(64, (4, 4), (2, 2), act=lrelu, name='E_conv_1')(I)
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
	E_net = Model(inputs=I, outputs=[z, mu, log_sigma], name='Encoder')
	return E_net
