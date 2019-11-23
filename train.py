import tensorflow as tf
import numpy as np
import model
import tensorlayer as tl
import loader
import glob
from model import Generator, Encoder, Discriminator
from loader import load_batch_image, load_test_image, load_images

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
opt = tf.optimizers.Adam(lrC, beta1)

load_images()
G.train()
D.train()
E.train()
train_A = len(glob.glob('cityscapes/train/*.jpg'))
num_batches = train_A // batch_size

image_A, image_B = load_batch_image(1)
image_A = np.expand_dims(image_A, axis=0)
image_B = np.expand_dims(image_B, axis=0)
image_A = image_A.astype(np.float32)
image_B = image_B.astype(np.float32)
z = np.random.normal(size=(batch_size, Z_dim))
z = z.astype(np.float32)
encoded_true_img, encoded_mu, encoded_log_sigma = E(image_B)
z = tf.reshape(z, z_shape0)
z = tf.tile(z, z_shape1)
encoded_true_img = tf.reshape(encoded_true_img, z_shape0)
encoded_true_img = tf.tile(encoded_true_img, z_shape1)
encoded_GI = tf.concat([image_A, encoded_true_img], axis=3)
GI = tf.concat([image_A, z], axis=3)
desired_gen_img = G(encoded_GI)
LR_desired_img = G(GI)

for epoch in range(epochs):
    for idx in range(train_A):
        image_A, image_B = load_batch_image(idx)
        image_A = np.expand_dims(image_A, axis=0)
        image_B = np.expand_dims(image_B, axis=0)
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

            loss_vae_D = (
				tl.cost.mean_squared_error(P_real, 0.9) +
                tl.cost.mean_squared_error(P_fake_encoded, 0))
            loss_lr_D = (
                tl.cost.mean_squared_error(P_real, 0.9) +
                tl.cost.mean_squared_error(P_fake, 0))
            loss_vae_GE = tl.cost.mean_squared_error(P_fake_encoded, 0.9)
            loss_G = tl.cost.mean_squared_error(P_fake, 0.9)
            loss_vae_GE = tf.metrics.mean_absolute_error(image_B, desired_gen_img)
            loss_latent_GE = tf.metrics.mean_absolute_error(z, reconst_z)
            loss_kl_E = 0.5 * tf.reduce_mean(
                -1 - encoded_log_sigma + encoded_mu**2 +
                tf.exp(encoded_log_sigma))

            loss_D = loss_vae_D + loss_lr_D - tl.cost.mean_squared_error(P_real, 0.9)
            loss_G = loss_vae_GE + loss_G + reconst_C * loss_vae_GE + latent_C * loss_latent_GE
            loss_E = loss_vae_GE + reconst_C * loss_vae_GE + latent_C * loss_latent_GE + kl_C * loss_kl_E

        grad = tape.gradient(loss_G, G.trainable_weights)
        opt.apply_gradients(zip(grad, G.trainable_weights))
        grad = tape.gradient(loss_D, D.trainable_weights)
        opt.apply_gradients(zip(grad, D.trainable_weights))
        grad = tape.gradient(loss_E, E.trainable_weights)
        opt.apply_gradients(zip(grad, E.trainable_weights))

    if epoch != 0 and epoch % 10 == 0:
        tl.vis.save_images(desired_gen_img.numpy(), [2, 4],
                           os.path.join(save_dir, 'vae_g_{}.png'.format(epoch)))
        tl.vis.save_images(LR_desired_img.numpy(), [2, 4],
                           os.path.join(save_dir, 'lr_g_{}.png'.format(epoch)))

#G.eval()
#out=G(valid_...).numpy()
