### Configuration file for experiments, trainer and NN models

from argparse import Namespace

def config_to_dict(ns: Namespace):
    return vars(ns)

def dict_to_config(d: dict):
    return Namespace(**d)

# Experiment parameters
project_entity = "matus13579"
project_name = "NN-z3"

# Trainer parameters
model_path = "bestNN.tar"

# Trainer hyperparameters
batch_size = 256
learning_rate = 0.01
betas = (0.9, 0.999)
weight_decay = 0.001
grad_clip = 10

# NN params
config_NN = Namespace (
    # Vocabulary parameters
    padding_index = 0, 

    # Embedding parameters
    embedding_dropout = 0.2, 
    vocab_size = 7054, 

    # RNN
    rnn_layers = 3, 
    bidirectional = False, 
    hidden_features = 256, 
    rnn_dropout = 0.2,
    momentum = 0.1,
    stepsize = 0.6, 
    rnn_beta = 0.2, 
)
