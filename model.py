import tensorflow as tf
import numpy as np
import tensorlayer as tl
from tensorlayer.layers import Input,Dense,Dropout,Conv2d,DeConv2d,BatchNorm2d,InstanceNorm2d,flatten_reshape,MeanPool2d
from tensorlayer.act import lrelu,tanh
from tensorlayer.models import Model

class BicycleGAN(object):
    model_name='BicycleGAN'

    def __init__(self,sess,epoch,batch_size,Z_dim,image_size,dataset_name,checkpoint_dir,result_dir,log_dir):
        self.sess=sess
        self.epoch=epoch
        self.batch_size=batch_size
        self.image_size=image_size
        self.Z_dim=Z_dim
        self.dataset_name=dataset_name
        self.checkpoint_dir=checkpoint_dir
        self.result_dir=result_dir
        self.log_dir=log_dir

        if dataset_name=='cityscapes'
            self.input_width=256
            self.input_height=256
            self.output_width=256
            self.output_height=256
            self.channels=3

            self.learning_rate=0.0002
            self.beta1=0.5
            self.beta2=0.999
            self.reconst_C=10
            self.latent_C=0.5
            self.kl_C=0.01

            self.sample_num=64

            self.train_A=glob.glob('city/scapes/train/*.jpg')

            self.num_batches=len(self.train_A)

    def residual(R,n_f,f_s):
        R_tmp=R
        R=BatchNorm2d(act=lrelu)(Conv2d(n_f,(f_s,f_s),(2,2))(R))
        R=BatchNorm2d(act=None)(Conv2d(n_f,(3,3),(1,1))(R))
        R_tmp=Conv2d(n_f,(1,1),(1,1))(R_tmp)
        return tf.nn.relu(R_tmp+R)

    def Discriminator(D):
        D=Conv2d(64,(4,4),(2,2),padding='SAME',act=lrelu,name='D_conv_1')(D)
        D=InstanceNorm2d(act=lrelu)(Conv2d(128,(4,4),(2,2),padding='SAME',name='D_conv_2')(D))
        D=InstanceNorm2d(act=lrelu)(Conv2d(256,(4,4),(2,2),padding='SAME',name='D_conv_3')(D))
        D=InstanceNorm2d(act=lrelu)(Conv2d(512,(4,4),(2,2),padding='SAME',name='D_conv_4')(D))
        D=InstanceNorm2d(act=lrelu)(Conv2d(512,(4,4),(2,2),padding='SAME',name='D_conv_5')(D))
        D=InstanceNorm2d(act=lrelu)(Conv2d(512,(4,4),(2,2),padding='SAME',name='D_conv_6')(D))
        D=Conv2d(1,(4,4),(2,2),name='D_conv_7')(D)
        D=tf.reduce_mean(D,axis=[1,2,3])
        return D

    def Generator(self,G,z):
        z=tf.reshape(z,[self.batch_size,1,1,self.Z_dim])
        z=tf.tile(z,[1,self.image_size,self.image_size,1])
        G=tf.concat([G,z],axis=3)
        G=Conv2d(64,(4,4),(2,2),padding='SAME',act=lrelu,name='G_conv_1')(I)
        G=BatchNorm2d(act=lrelu)(Conv2d(128,(4,4),(2,2),padding='SAME',name='G_conv_2')(G))
        G=BatchNorm2d(act=lrelu)(Conv2d(256,(4,4),(2,2),padding='SAME',name='G_conv_3')(G))
        G=BatchNorm2d(act=lrelu)(Conv2d(512,(4,4),(2,2),padding='SAME',name='G_conv_4')(G))
        G=BatchNorm2d(act=lrelu)(Conv2d(512,(4,4),(2,2),padding='SAME',name='G_conv_5')(G))
        G=BatchNorm2d(act=lrelu)(Conv2d(512,(4,4),(2,2),padding='SAME',name='G_conv_6')(G))
        G=BatchNorm2d(act=lrelu)(Conv2d(512,(4,4),(2,2),padding='SAME',name='G_conv_7')(G))
        G=BatchNorm2d(act=lrelu)(Conv2d(512,(4,4),(2,2),padding='SAME',name='G_conv_8')(G))
        G=BatchNorm2d(act=lrelu)(DeConv2d(512,(4,4),(2,2),padding='SAME',name='G_deconv_1')(G))
        G=BatchNorm2d(act=lrelu)(DeConv2d(512,(4,4),(2,2),padding='SAME',name='G_deconv_2')(G))
        G=BatchNorm2d(act=lrelu)(DeConv2d(512,(4,4),(2,2),padding='SAME',name='G_deconv_3')(G))
        G=BatchNorm2d(act=lrelu)(DeConv2d(512,(4,4),(2,2),padding='SAME',name='G_deconv_4')(G))
        G=BatchNorm2d(act=lrelu)(DeConv2d(256,(4,4),(2,2),padding='SAME',name='G_deconv_5')(G))
        G=BatchNorm2d(act=lrelu)(DeConv2d(128,(4,4),(2,2),padding='SAME',name='G_deconv_6')(G))
        G=BatchNorm2d(act=lrelu)(DeConv2d(64,(4,4),(2,2),padding='SAME',name='G_deconv_7')(G))
        G=BatchNorm2d(act=tanh)(DeConv2d(3,(4,4),(2,2),padding='SAME',name='G_deconv_8')(G))
        return D

    def encoder(self,E):
        E=Conv2d(64,(4,4),(2,2),act=lrelu,name='E_conv_1')(E)
        E=MeanPool2d((2,2),(2,2),'SAME')(self.residual(E,128,3))
        E=MeanPool2d((2,2),(2,2),'SAME')(self.residual(E,256,3))
        E=MeanPool2d((2,2),(2,2),'SAME')(self.residual(E,512,3))
        E=MeanPool2d((2,2),(2,2),'SAME')(self.residual(E,512,3))
        E=MeanPool2d((2,2),(2,2),'SAME')(self.residual(E,512,3))
        E=MeanPool2d((8,8),(8,8),'SAME')(E)
        mu=Dense(self.Z_dim)(E)
        log_sigma=Dense(self.Z_dim)(E)
        z=mu+tf.random_normal(shape=tf.shape(self.Z_dim))*tf.exp(log_sigma)
        return z,mu,log_sigma

    def build_model(self):
        image_dims=[self.input_width,self.input_height,self.channels]

        self.image_A=
        self.image_B=
        self.z=

        self.encoded_true_img,self.encoded_mu,self.encoded_log_sigma=self.Encoder(self.image_B)

        self.desired_gen_img=self.Generator(self.image_A,self.encoded_true_img)

        self.LR_desired_img=self.Generator(self.image_A,self.z)

        self.reconst_z,self.reconst_mu,self.reconst_log_sigma=self.Encoder(self.LR_desired_img)

        self.P_real=self.Discriminator(self.image_B)
        self.P_fake=self.Discriminator(self.LR_desired_img)
        self.P_fake_encoded=self.Discriminator(self.desired_gen_img)

	self.loss_vae_D=(tf.reduce_mean(tf.squared_difference(self.P_real,0.9))+tf.reduce_mean(tf.square(self.P_fake_encoded)))
	self.loss_lr_D=(tf.reduce_mean(tf.squared_difference(self.P_real,0.9))+tf.reduce_mean(tf.square(self.P_fake)))	
	self.loss_vae_GE=tf.reduce_mean(tf.squared_difference(self.P_fake_encoded,0.9))
	self.loss_G=tf.reduce_mean(tf.squared_difference(self.P_fake,0.9))
	self.loss_vae_GE=tf.reduce_mean(tf.abs(self.image_B-self.desired_gen_img))
	self.loss_latent_GE=tf.reduce_mean(tf.abs(self.z-self.reconst_z))
	self.loss_kl_E=0.5*tf.reduce_mean(-1-self.encoded_log_sigma+self.encoded_mu**2+tf.exp(self.encoded_log_sigma))
        
        self.loss_D=self.loss_vae_gan_D+self.loss_lr_gan_D-tf.reduce_mean(tf.squared_difference(self.P_real, 0.9))
        self.loss_G=self.loss_vae_gan_GE+self.loss_gan_G+self.reconst_C*self.loss_vae_GE+self.latent_C*self.loss_latent_GE
        self.loss_E=self.loss_vae_gan_GE+self.reconst_C*self.loss_vae_GE+self.latent_C*self.loss_latent_GE+self.kl_C*self.loss_kl_E



    def train(self):


    def test(self):

    def model_dir(self):

    def save(self,checkpoint_dir,step):


    def load(self,checkpoint_dir):
        
