### Configuration file for experiments, trainer and NN models

from argparse import Namespace

def config_to_dict(ns: Namespace):
    return vars(ns)

def dict_to_config(d: dict):
    return Namespace(**d)

# Experiment parameters
project_entity = "matus13579"
project_name = "NN-z3"
#data_dir = "./data_norm/"


# Trainer parameters
# TODO metric names 

model_path = "bestNN.tar"
save_interval = 1

batch_size = 8
learning_rate = 0.001
betas = (0.9, 0.999)
weight_decay = 0.0001

# U-Net params
config_Unet = Namespace (
    hidden_layers = 50
)
