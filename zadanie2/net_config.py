from argparse import Namespace

print("Network configuration loaded")

def config_to_dict(ns: Namespace):
    return vars(ns)

def dict_to_config(d: dict):
    return Namespace(**d)

config_Unet = Namespace (
    # Dataset parameters
    batch_size = 64,

    # Trainer params
    learning_rate = 0.001,
    betas = (0.9, 0.999), 
    weight_decay = 0, 
    initial_bias = 0,

    # U-Net params
    channels_out_init = 32,             # initial channel width (num of channels on 1-st layer)
    channel_mul = float(2),             # multiplication of image channels per layer
    network_depth = 3,                  # number of network layers (without bridge)
    skip_features = "concat",           # none | concat | sdi    

    # ENC, DEC params
    # h,w should be consistent per layer (otherwise output has different h,w then input)
    block_width = 2,            
    kernel_size = 5,                    # affects h,w 
    padding = 2,                        # affects h,w 
    stride = 1,                         # affects h,w 
    dilation = 1,                       # affects h,w 
    pool_type = "max",                  # max | avg 
    pool_kernel_size = 2,               # affects h,w (pooling and transposed convolution)
    padding_convT = [1, 0, 0, 0]        # needed if h,w is not divisible by at i-layer (1 = not divisible)
)

a = 5
