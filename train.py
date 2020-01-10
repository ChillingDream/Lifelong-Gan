from time import sleep

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
	tf.config.experimental.set_visible_devices(gpus[arg.gpu], 'GPU')
	logical_gpus = tf.config.experimental.list_logical_devices('GPU')
	print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

tl.files.exists_or_mkdir(sample_dir)
tl.files.exists_or_mkdir(models_dir)
tl.files.exists_or_mkdir(log_dir)

writer = tf.summary.create_file_writer(log_dir)

bicycleGAN = BicycleGAN(LOAD, load_tag)
preGAN = BicycleGAN(mode == "incremental", load_tag)
global_step = 0

def train_one_task(train_data, task = "", use_aux_data = False):
	global global_step
	G_lr = tf.Variable(initial_lr, dtype=tf.float32, name="G_learning_rate")
	D_lr = tf.Variable(initial_lr, dtype=tf.float32, name="D_learning_rate")
	E_lr = tf.Variable(initial_lr, dtype=tf.float32, name="E_learning_rate")
	G_optimizer = tf.optimizers.Adam(G_lr, beta1)
	D_optimizer = tf.optimizers.RMSprop(D_lr, beta1)
	E_optimizer = tf.optimizers.Adam(E_lr, beta1)

	epoch = 0
	num_images = len(train_data)
	t = trange(0, num_images * epochs, batch_size)

	for step, _ in zip(t, train_data(batch_size, shuffle=False, use_aux=use_aux_data)):
		global_step += 1
		if use_aux_data:
			image_A, image_B, aux_A, aux_B = _
		else:
			image_A, image_B = _
		z = tf.random.normal(shape=(batch_size, z_dim))
		with tf.GradientTape(persistent=True) as tape:
			loss = bicycleGAN.calc_loss(image_A, image_B, z)
			if use_aux_data:
				encoded_z = bicycleGAN.E(aux_B)[0]
				pre_encoded_z = preGAN.E(aux_B)[0]
				vae_dl = tl.cost.absolute_difference_error(encoded_z, pre_encoded_z, is_mean=True) + \
					tl.cost.absolute_difference_error(bicycleGAN.G([aux_A, encoded_z]), preGAN.G([aux_A, pre_encoded_z]), is_mean=True)

				aux_z = tf.random.normal(shape=(batch_size, z_dim))
				lr_img = bicycleGAN.G([aux_A, aux_z])
				pre_lr_img = preGAN.G([aux_A, aux_z])
				lr_dl = tl.cost.absolute_difference_error(lr_img, pre_lr_img, is_mean=True) + \
					tl.cost.absolute_difference_error(bicycleGAN.E(lr_img)[0], preGAN.E(pre_lr_img)[0])

				loss_dl = vae_dl + lr_dl
				loss -= dl_beta * loss_dl
			nloss = -loss
		grad = tape.gradient(loss, bicycleGAN.D.trainable_weights)
		#grad, norm_D = tf.clip_by_global_norm(grad, 10)
		#tf.summary.scalar("gradients_norm/norm_D", norm_D, step)
		#D_optimizer.apply_gradients(zip(grad, bicycleGAN.D.trainable_weights))

		grad = tape.gradient(nloss, bicycleGAN.G.trainable_weights)
		#grad, norm_G = tf.clip_by_global_norm(grad, 10)
		#tf.summary.scalar("gradients_norm/norm_G", norm_G, step)
		#G_optimizer.apply_gradients(zip(grad, bicycleGAN.G.trainable_weights))

		grad = tape.gradient(nloss, bicycleGAN.E.trainable_weights)
		#grad, norm_E = tf.clip_by_global_norm(grad, 10)
		#tf.summary.scalar("gradients_norm/norm_E", norm_E, step)
		#E_optimizer.apply_gradients(zip(grad, bicycleGAN.E.trainable_weights))

		del tape
		t.set_description("%f" % (bicycleGAN.loss_vae_L1))
		tf.summary.scalar("loss/loss", loss, global_step)
		tf.summary.scalar("loss/loss_GAN_G", bicycleGAN.loss_G, global_step)
		tf.summary.scalar("loss/loss_D", bicycleGAN.loss_D, global_step)
		tf.summary.scalar("loss/loss_vae_L1", bicycleGAN.loss_vae_L1, global_step)
		tf.summary.scalar("loss/loss_latent_L1", bicycleGAN.loss_latent_L1, global_step)
		tf.summary.scalar("loss/loss_kl_E", bicycleGAN.loss_kl_E, global_step)
		tf.summary.scalar("model/P_real", bicycleGAN.P_real[0][0], global_step)
		tf.summary.scalar("model/P_fake", bicycleGAN.P_fake[0][0], global_step)
		tf.summary.scalar("model/P_fake_encoded", bicycleGAN.P_fake_encoded[0][0], global_step)
		if use_aux_data:
			tf.summary.scalar("loss/loss_dl", loss_dl, global_step)

		if step % log_step == log_step - 1:
			tl.vis.save_images(bicycleGAN.vae_img.numpy(), [1, batch_size], os.path.join(sample_dir, 'vae_{}{}.png'.format(task, global_step)))
			tl.vis.save_images(bicycleGAN.lr_img.numpy(), [1, batch_size], os.path.join(sample_dir, 'lr_{}{}.png'.format(task, global_step)))

		if step % num_images == num_images - 1:
			bicycleGAN.save(model_tag)
			epoch += 1
			#new_G_lr = initial_lr * (lr_decay_G ** epoch)
			#new_D_lr = initial_lr * (lr_decay_D ** epoch) / (1 + epoch)
			#G_lr.assign(new_G_lr)
			#D_lr.assign(new_D_lr)
			#E_lr.assign(new_G_lr)
			tf.summary.scalar("learing_rate/lr_G", G_lr, global_step)
			tf.summary.scalar("learing_rate/lr_D", D_lr, global_step)

		writer.flush()

if mode == "continual" or "incremental":
	print("{} tasks in total.".format(len(tasks)))
	for i, task in enumerate(tasks):
		print("Task {} ...".format(i + 1))
		train_data = DataGenerator(task, "train")
		with tf.device("/gpu:0"), writer.as_default():
			if i > 0:
				preGAN.load(model_tag)
			train_one_task(train_data, task, i > 0 or mode == "incremental")
		sleep(1)
else:
	print("Joint training ...")
	with tf.device("/gpu:0"), writer.as_default():
		train_data = DataGenerator(tasks, "train")
		train_one_task(train_data, use_aux_data=False)
print("Training finishes.")
