import argparse

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument("--mode", default="train", type=str, help="train or test")
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--image_size", default=256, type=int)
parser.add_argument("--tag", default="default", type=str)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--dataset", default="facades&edges2shoes")
parser.add_argument("--log_dir", default="logs/", type=str)

arg = parser.parse_args()

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
reconst_C = 10
latent_C = 0.5
kl_C = 0.01
save_dir = 'samples/'
models_dir = 'nets/'
log_dir = arg.log_dir