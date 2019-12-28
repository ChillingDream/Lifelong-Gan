import argparse

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument("--mode", default="continual", type=str, help="continual or joint")
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--image_size", default=256, type=int)
parser.add_argument("--tag", default="default", type=str)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--log_dir", default="logs/", type=str)
parser.add_argument("--model_dir", default="nets/", type=str)
parser.add_argument("--save_dir", default="samples/", type=str)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--max_image_num", default=5000, type=int)
parser.add_argument("--tasks", default="facades", type=str)
parser.add_argument("--load", default=False, action="store_true")

arg = parser.parse_args()

mode = arg.mode
batch_size = arg.batch_size
model_tag = arg.tag
epochs = arg.epochs
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
kl_C = 0.01
save_dir = arg.save_dir
models_dir = arg.model_dir
log_dir = arg.log_dir
max_image_num = arg.max_image_num
tasks = arg.tasks.split(sep='+')
LOAD = arg.load