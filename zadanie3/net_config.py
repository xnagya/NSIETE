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

# Trainer hyperparameters
batch_size = 64
learning_rate = 0.001
betas = (0.9, 0.999)
weight_decay = 0.0001

# NN params
config_NN = Namespace (
    # Vocabulary parameters
    padding_index = 0, 

    # RNN
    embedding_dropout = 0, 
    lstm_layers = 3, 
    bidirectional = False, 

    # LSTM
    lstm_features = 64, 
    lstm_dropout = 0,
)
