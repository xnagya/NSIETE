### Configuration file for experiments, trainer and NN models

from argparse import Namespace

def config_to_dict(ns: Namespace):
    return vars(ns)

def dict_to_config(d: dict):
    return Namespace(**d)

num_of_classes = 7
background_class = 0

# Experiment parameters
project_entity = "matus13579"
project_name = "NN-z2"
data_dir = "./data_norm/"


# Trainer parameters
metric_name_Tloss = "train_loss"
metric_name_Vloss = "val_loss"
metric_name_acc = "accuracy"
metric_name_iou = "iou"
metric_name_dice = "dice"
batch_size = 16
learning_rate = 0.01
betas = (0.9, 0.999)
weight_decay = 0.0001
model_path = "UNet-v2.tar"
save_interval = 3


# U-Net params
config_Unet = Namespace (
    channels_out_init = 16,             # initial channel width (num of output channels on 1-st layer)
    channel_mul = float(2),             # multiplication of image channels per layer, idealy int number
    network_depth = 3,                  # number of network layers (without bridge)
    skip_features = "concat",           # none | concat     

    # ENC, DEC params
    # h,w should be consistent per layer (otherwise output has different h,w then input)
    block_width = 2,            
    kernel_size = 5,                    # affects h,w 
    padding = 2,                        # affects h,w 
    stride = 1,                         # affects h,w 
    dilation = 1,                       # affects h,w 
    pool_type = "max",                  # max | avg 
    pool_kernel_size = 3,               # affects h,w (pooling and transposed convolution)
    padding_convT = [1, 1, 2],          # needed when h,w is not divisible by at i-layer (1 = not divisible)

    channel_att_ratio = 16,
    SDI_channels = 32
)
