import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Input, Dense, Conv2d, DeConv2d, BatchNorm2d, InstanceNorm2d, Flatten, MeanPool2d, Elementwise, Lambda
from tensorlayer.models import Model

lrelu = lambda x: tl.act.lrelu(x, 0.2)


def residual(R, n_f, f_s):
	w_init = tl.initializers.truncated_normal(stddev=0.001)
	R_tmp = R
	R = BatchNorm2d(act=lrelu)(Conv2d(n_f, (f_s, f_s), (1, 1), W_init=w_init)(R))
	R = BatchNorm2d(act=None)(Conv2d(n_f, (f_s, f_s), (1, 1), W_init=w_init)(R))
	R_tmp = Conv2d(n_f, (1, 1), (1, 1))(R_tmp)
	return Elementwise(tf.add, act=tf.nn.relu)([R_tmp, R])


def Discriminator(input_shape):
	I = Input(input_shape)
	D = Conv2d(
		64, (4, 4), (2, 2), padding='SAME', act=lrelu, name='D_conv_1')(I)
	D = InstanceNorm2d(act=lrelu)(Conv2d(
		128, (4, 4), (2, 2), padding='SAME', name='D_conv_2')(D))
	D = InstanceNorm2d(act=lrelu)(Conv2d(
		256, (4, 4), (2, 2), padding='SAME', name='D_conv_3')(D))
	D = InstanceNorm2d(act=lrelu)(Conv2d(
		512, (4, 4), (2, 2), padding='SAME', name='D_conv_4')(D))
	D = InstanceNorm2d(act=lrelu)(Conv2d(
		512, (4, 4), (2, 2), padding='SAME', name='D_conv_5')(D))
	D = InstanceNorm2d(act=lrelu)(Conv2d(
		512, (4, 4), (2, 2), padding='SAME', name='D_conv_6')(D))
	D = Conv2d(1, (4, 4), (2, 2), name='D_conv_7', b_init=0.0)(D)
	D = Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2, 3]))(D)
	D_net = Model(inputs=I, outputs=D, name='Discriminator')
	return D_net


def Generator(input_shape):
	w_init = tl.initializers.truncated_normal(stddev=0.001)
	I = Input(input_shape)
	G = Conv2d(
		64, (4, 4), (2, 2), padding='SAME', act=lrelu, W_init=w_init, name='G_conv_1')(I)
	G = BatchNorm2d(act=lrelu)(Conv2d(
		128, (4, 4), (2, 2), padding='SAME', W_init=w_init, name='G_conv_2')(G))
	G = BatchNorm2d(act=lrelu)(Conv2d(
		256, (4, 4), (2, 2), padding='SAME', W_init=w_init, name='G_conv_3')(G))
	G = BatchNorm2d(act=lrelu)(Conv2d(
		512, (4, 4), (2, 2), padding='SAME', W_init=w_init, name='G_conv_4')(G))
	G = BatchNorm2d(act=lrelu)(Conv2d(
		512, (4, 4), (2, 2), padding='SAME', W_init=w_init, name='G_conv_5')(G))
	G = BatchNorm2d(act=lrelu)(Conv2d(
		512, (4, 4), (2, 2), padding='SAME', W_init=w_init, name='G_conv_6')(G))
	G = BatchNorm2d(act=lrelu)(Conv2d(
		512, (4, 4), (2, 2), padding='SAME', W_init=w_init, name='G_conv_7')(G))
	G = BatchNorm2d(act=lrelu)(Conv2d(
		512, (4, 4), (2, 2), padding='SAME', W_init=w_init, name='G_conv_8')(G))
	G = BatchNorm2d(act=lrelu)(DeConv2d(
		512, (4, 4), (2, 2), padding='SAME', W_init=w_init, name='G_deconv_1')(G))
	G = BatchNorm2d(act=lrelu)(DeConv2d(
		512, (4, 4), (2, 2), padding='SAME', W_init=w_init, name='G_deconv_2')(G))
	G = BatchNorm2d(act=lrelu)(DeConv2d(
		512, (4, 4), (2, 2), padding='SAME', W_init=w_init, name='G_deconv_3')(G))
	G = BatchNorm2d(act=lrelu)(DeConv2d(
		512, (4, 4), (2, 2), padding='SAME', W_init=w_init, name='G_deconv_4')(G))
	G = BatchNorm2d(act=lrelu)(DeConv2d(
		256, (4, 4), (2, 2), padding='SAME', W_init=w_init, name='G_deconv_5')(G))
	G = BatchNorm2d(act=lrelu)(DeConv2d(
		128, (4, 4), (2, 2), padding='SAME', W_init=w_init, name='G_deconv_6')(G))
	G = BatchNorm2d(act=lrelu)(DeConv2d(
		64, (4, 4), (2, 2), padding='SAME', W_init=w_init, name='G_deconv_7')(G))
	G = BatchNorm2d(act=tf.nn.tanh)(DeConv2d(
		3, (4, 4), (2, 2), padding='SAME', W_init=w_init, name='G_deconv_8')(G))
	G_net = Model(inputs=I, outputs=G, name='Generator')
	return G_net


def Encoder(input_shape, Z_dim):
	I = Input(input_shape)
	E = Conv2d(64, (4, 4), (2, 2), act=lrelu, name='E_conv_1')(I)
	E = MeanPool2d((2, 2), (2, 2), 'SAME')(residual(E, 128, 3))
	E = MeanPool2d((2, 2), (2, 2), 'SAME')(residual(E, 256, 3))
	E = MeanPool2d((2, 2), (2, 2), 'SAME')(residual(E, 512, 3))
	E = MeanPool2d((2, 2), (2, 2), 'SAME')(residual(E, 512, 3))
	E = MeanPool2d((2, 2), (2, 2), 'SAME')(residual(E, 512, 3))
	E = Flatten()(MeanPool2d((8, 8), (8, 8), 'SAME')(E))
	mu = Dense(Z_dim, b_init=0)(E)
	log_sigma = Dense(Z_dim, b_init=0)(E)
	z = Elementwise(tf.add)(
		[mu, Lambda(lambda x: tf.random.normal(shape=[Z_dim]) * tf.exp(x))(log_sigma)])
	E_net = Model(inputs=I, outputs=[z, mu, log_sigma], name='Encoder')
	return E_net
