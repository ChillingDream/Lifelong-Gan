import tensorflow as tf
import time
import numpy as np
import tensorlayer as tl
import os
from model import Generator, Encoder, Discriminator
from loader import load_images

epochs = 20
batch_size = 1
image_size = 256
Z_dim = 8
dataset_name = 'cityscapes'
save_dir = 'samples'
tl.files.exists_or_mkdir(save_dir)
#sample_num=64
#log_dir
#check_point_dir
#result_dir
lrC = 0.0002  #learning rate
beta1 = 0.5
#beta2=0.999
reconst_C = 10
latent_C = 0.5
kl_C = 0.0001
image_dims = [256, 256, 3]
z_shape0 = (batch_size, 1, 1, Z_dim)
z_shape1 = (1, image_size, image_size, 1)
G = Generator((batch_size, 256, 256, 3 + Z_dim))
D = Discriminator((batch_size, 256, 256, 3))
E = Encoder((batch_size, 256, 256, 3), Z_dim)
G_optimizer = tf.optimizers.Adam(lrC, beta1)
D_optimizer = tf.optimizers.Adam(lrC, beta1)
E_optimizer = tf.optimizers.Adam(lrC, beta1)

G.train()
D.train()
E.train()
train_cityscapes = load_images("cityscapes", "train")

def train_one_task(train_data, use_aux_data = False):
	start_time = time.time()
	for epoch in range(epochs):
		tot_loss = []
		processed = 0
		for image_A, image_B in tl.iterate.minibatches(train_data[0], train_data[1], batch_size, shuffle=False):
			z = np.random.normal(size=(batch_size, Z_dim))
			image_A = image_A.astype(np.float32)
			image_B = image_B.astype(np.float32)
			z = z.astype(np.float32)

			with tf.GradientTape(persistent=True) as tape:
				encoded_true_img, encoded_mu, encoded_log_sigma = E(image_B)

				z = tf.reshape(z, z_shape0)
				z = tf.tile(z, z_shape1)
				encoded_true_img = tf.reshape(encoded_true_img, z_shape0)
				encoded_true_img = tf.tile(encoded_true_img, z_shape1)
				encoded_GI = tf.concat([image_A, encoded_true_img], axis=3)
				GI = tf.concat([image_A, z], axis=3)
				desired_gen_img = G(encoded_GI)
				LR_desired_img = G(GI)

				reconst_z, reconst_mu, reconst_log_sigma = E(LR_desired_img)

				P_real = D(image_B)
				P_fake = D(LR_desired_img)
				P_fake_encoded = D(desired_gen_img)

				loss_vae_D = (tl.cost.sigmoid_cross_entropy(P_real, tf.ones_like(P_real)) +
							  tl.cost.sigmoid_cross_entropy(P_fake_encoded, tf.zeros_like(P_fake_encoded)))
				loss_lr_D = (tl.cost.sigmoid_cross_entropy(P_real, tf.ones_like(P_real)) +
							 tl.cost.sigmoid_cross_entropy(P_fake, tf.zeros_like(P_fake)))
				loss_vae_G = tl.cost.sigmoid_cross_entropy(P_fake_encoded, tf.ones_like(P_fake_encoded))
				loss_G = tl.cost.sigmoid_cross_entropy(P_fake, tf.ones_like(P_fake))
				loss_vae_L1 = tl.cost.absolute_difference_error(
					image_B, desired_gen_img, axis=[-1, -2])
				loss_latent_GE = tl.cost.absolute_difference_error(z, reconst_z)
				loss_kl_E = 0.5 * tf.reduce_mean(
					-1 - 2 * encoded_log_sigma + encoded_mu**2 +
					tf.exp(2 * encoded_log_sigma))

				loss_D = loss_vae_D + loss_lr_D
				loss_G = loss_vae_G + loss_G + reconst_C * loss_vae_L1 + latent_C * loss_latent_GE
				loss_E = loss_vae_G + reconst_C * loss_vae_L1 + latent_C * loss_latent_GE + kl_C * loss_kl_E
				tot_loss.append([loss_G, loss_D, loss_E])

			grad = tape.gradient(loss_G, G.trainable_weights)
			grad, _ = tf.clip_by_global_norm(grad, 5)
			G_optimizer.apply_gradients(zip(grad, G.trainable_weights))
			grad = tape.gradient(loss_D, D.trainable_weights)
			grad, _ = tf.clip_by_global_norm(grad, 5)
			D_optimizer.apply_gradients(zip(grad, D.trainable_weights))
			grad = tape.gradient(loss_E, E.trainable_weights)
			grad, _ = tf.clip_by_global_norm(grad, 5)
			E_optimizer.apply_gradients(zip(grad, E.trainable_weights))
			del tape
			processed += batch_size
			if processed > 100:
				print("#", end="")
				processed -= 100

		print("")
		if epoch % 1 == 0:
			tot_loss = np.mean(tot_loss, 0)
			print("time_used=%f" % (time.time() - start_time))
			print("Epoch %d finished! loss_G=%f loss_D=%f loss_E=%f" % (epoch, tot_loss[0], tot_loss[1], tot_loss[2]))
			tl.vis.save_images(desired_gen_img.numpy(), [1, 1],
							   os.path.join(save_dir,
											'vae_g_{}.png'.format(epoch)))
			tl.vis.save_images(LR_desired_img.numpy(), [1, 1],
							   os.path.join(save_dir,
											'lr_g_{}.png'.format(epoch)))

train_one_task(train_cityscapes, None)

#G.eval()
#out=G(valid_...).numpy()
