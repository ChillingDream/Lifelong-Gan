import argparse
import os

import tensorflow as tf

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument("--mode", default="continual", type=str, help="continual, incremental or joint")
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--image_size", default=256, type=int)
parser.add_argument("--tag", default="default", type=str)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--load_tag", default="", type=str)
parser.add_argument("--log_dir", default="logs/", type=str)
parser.add_argument("--model_dir", default="nets/", type=str)
parser.add_argument("--sample_dir", default="samples/", type=str)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--max_image_num", default=5000, type=int)
parser.add_argument("--tasks", default="facades", type=str)
parser.add_argument("--log_step", default=400, type=int)
parser.add_argument("--patch_size", default=64, type=int)

arg = parser.parse_args()

mode = arg.mode
batch_size = arg.batch_size
model_tag = arg.tag
epochs = arg.epochs
decay_epochs = 4
image_size = arg.image_size
z_dim = 8
image_shape = (image_size, image_size, 3)
input_shape = (batch_size, image_size, image_size, 3)
initial_lr = 0.0002
lr_decay_G = 0.9
lr_decay_D = 0.85
beta1 = 0.5
dl_beta = 5.
reconst_C = 10
latent_C = 0.5
kl_C = 0.05
sample_dir = os.path.join(arg.sample_dir, model_tag)
models_dir = os.path.join(arg.model_dir, model_tag)
log_dir = os.path.join(arg.log_dir, model_tag)
max_image_num = arg.max_image_num
tasks = arg.tasks.split(sep='+')
LOAD = (mode == "incremental")
if arg.load_tag:
	LOAD = True
	load_tag = arg.load_tag
else:
	load_tag = model_tag
log_step = arg.log_step
patch_size = arg.patch_size

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)
	tf.config.experimental.set_visible_devices(gpus[arg.gpu], 'GPU')
	logical_gpus = tf.config.experimental.list_logical_devices('GPU')
	print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

