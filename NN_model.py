# %% [markdown]
# # __Neural Network (NN) model implementation__

# %%
from collections import OrderedDict
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from argparse import Namespace

# %% [markdown]
# ### __Configuration of NN__ (Namespace variable)

# %%
config = Namespace(
    # Dataset parameters
    batch_size = 128,

    # Model parameters
    learning_rate = 0.001,
    normalized_weight_init = False,
    initial_bias = 0, 
    activation_fn = "sigmoid", # sigmoid | tanh | softsign | optimal
    neurons_hidden1 = 50,
    neurons_hidden2 = 50,
    neurons_hidden3 = 50,
    neurons_hidden4 = 50
)

def config_to_dict(ns: Namespace):
    return vars(ns)

def dict_to_config(d: dict):
    return Namespace(**d)

def normalize_dataset():
    if (config.activation_fn == "tanh" or config.activation_fn == "softsign"): 
        return True
    
    return False

# %% [markdown]
# ### __Metrics__
# This class represnts all data collected during one training epoch. They are used to evaluate our NN model. This data is implmented as a class for easier integration with wandb.
# ####  Metrics used:
# > Accuracy <br>
# > F1 Score

# %%
class Metrics:
    def __init__(self, y_true, y_pred):
        # Transorm predictions to classes
        if normalize_dataset():
            class_pred = binary_cutoff(y_pred, -1, 0,  1)
        else:
            class_pred = binary_cutoff(y_pred, 0, 0.5, 1)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, class_pred).ravel()
        
        #print (f"tn = {tn}")
        #print (f"fp = {fp}")
        #print (f"fn = {fn}")
        #print (f"tp = {tp}")
        #print(y_true)
        #print(y_pred)
        #print(class_pred)

        # Calculate metrics
        self.accuracy = (tp + tn) / (tp + tn + fp + fn)
        self.precision = tp / (tp + fp)
        self.recall = tp / (tp + fn)
        self.f1_score = (2 * self.precision * self.recall) / (self.precision + self.recall)

        self.accuracy = round(self.accuracy, 4) 
        self.precision = round(self.precision, 4) 
        self.recall = round(self.recall, 4) 
        self.f1_score = round(self.f1_score, 4) 

def binary_cutoff(predicted, class1, cutoff, class2):
    actual_predictions = []

    for y in predicted:
        if (y < cutoff):
            actual_predictions.append(class1)
        else:
            actual_predictions.append(class2)

    return actual_predictions

# %% [markdown]
# ### __FeedForward NN model__
# This class represents multilayer perceptron of feedforward NN used for binary classification on __Bioresponse__ dataset. It is stored as a variable in class __Trainer__ and contains functions for intilization of NN layers and intilization of their weight and biases. <br><br>
# > __Number of hidden layers:__ 4 <br>
# > __Activation between layers:__ sigmoid OR tanh OR softsign | in config (_activation_fn_)<br>
# > __Learinig rate:__ in config (_learning_rate_)<br>
# > __Intiliazation (weights):__ (_normalized_weight_init_ = TRUE) Xavier uniform distribution, (_normalized_weight_init_ = FALSE) uniform distribution<br>
# > __Intiliazation (bias):__ in config (_initial_bias_)

# %%
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # create first hidden layer
        layers = OrderedDict([
            ("hidden1", nn.Linear(input_size, config.neurons_hidden1, dtype = torch.float64))
        ])
        
        # Add other hidden + output layers with activation function
        match config.activation_fn:
            case "sigmoid":
                layers.update({"sig1" : nn.Sigmoid()})
                layers.update({"hidden2" : nn.Linear(config.neurons_hidden1, config.neurons_hidden2, dtype = torch.float64)})
                layers.update({"sig2" : nn.Sigmoid()})
                layers.update({"hidden3" : nn.Linear(config.neurons_hidden2, config.neurons_hidden3, dtype = torch.float64)})
                layers.update({"sig3" : nn.Sigmoid()})
                layers.update({"hidden4" : nn.Linear(config.neurons_hidden3, config.neurons_hidden4, dtype = torch.float64)})
                layers.update({"sig4" : nn.Sigmoid()})

            case "tanh":
                layers.update({"tanh1" : nn.Tanh()})
                layers.update({"hidden2" : nn.Linear(config.neurons_hidden1, config.neurons_hidden2, dtype = torch.float64)})
                layers.update({"tanh2" : nn.Tanh()})
                layers.update({"hidden3" : nn.Linear(config.neurons_hidden2, config.neurons_hidden3, dtype = torch.float64)})
                layers.update({"tanh3" : nn.Tanh()})
                layers.update({"hidden4" : nn.Linear(config.neurons_hidden3, config.neurons_hidden4, dtype = torch.float64)})
                layers.update({"tanh4" : nn.Tanh()})

            case "softsign":
                layers.update({"softs1" : nn.Softsign()})
                layers.update({"hidden2" : nn.Linear(config.neurons_hidden1, config.neurons_hidden2, dtype = torch.float64)})
                layers.update({"softs2" : nn.Softsign()})
                layers.update({"hidden3" : nn.Linear(config.neurons_hidden2, config.neurons_hidden3, dtype = torch.float64)})
                layers.update({"softs3" : nn.Softsign()})
                layers.update({"hidden4" : nn.Linear(config.neurons_hidden3, config.neurons_hidden4, dtype = torch.float64)})
                layers.update({"softs4" : nn.Softsign()})
                
            case _ :
                if (config.activation_fn != "optimal"):
                    print(f"ERROR: Wrong NN configuration: activation function = {config.activation_fn}")

                layers.update({"relu1" : nn.ReLU()})
                layers.update({"hidden2" : nn.Linear(config.neurons_hidden1, config.neurons_hidden2, dtype = torch.float64)})
                layers.update({"relu2" : nn.ReLU()})
                layers.update({"hidden3" : nn.Linear(config.neurons_hidden2, config.neurons_hidden3, dtype = torch.float64)})
                layers.update({"relu3" : nn.ReLU()})
                layers.update({"hidden4" : nn.Linear(config.neurons_hidden3, config.neurons_hidden4, dtype = torch.float64)})
                layers.update({"relu4" : nn.ReLU()})

        # Output 
        layers.update({"output" : nn.Linear(config.neurons_hidden4, 1, dtype = torch.float64)})
        layers.update({"sig_OUT" : nn.Sigmoid()})

        self.network = nn.Sequential(layers)
        self.init_weights()
        
    def init_weights(self):
        # Xavier uniform distribution
        if config.normalized_weight_init:
            for m in self.network.children():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(config.initial_bias)

        # Uniform distribution
        else:
            for m in self.network.children():
                if isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight)
                    m.bias.data.fill_(config.initial_bias)
    
    def forward(self,x):
        y = self.network(x)
        return y

# %%
class Trainer:
    def __init__(self, config: Namespace, model: MultiLayerPerceptron):
        self.cfg = config

        # Select GPU device
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device for training")

        # Move model to available device
        self.model = model.to(self.device)

        # Optimizer - Stochastic gradient descent
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.learning_rate) # momentum=0.9

        # Loss function - Binary Cross Entropy
        self.loss_fn = nn.BCELoss()

        self.loss_train = []
        self.loss_validate = []

    # Create Data Loaders
    def load_dataset(self, train_data, test_data):
        self.train_data = DataLoader(train_data, batch_size=self.cfg.batch_size, shuffle=True)
        self.test_data = DataLoader(test_data, batch_size=self.cfg.batch_size, shuffle=True)

    def train(self, logger = None):
        self.model.train()
        self.loss_train = []

        # Train model on each dataset batch (train_data)
        for batch, (x, y) in enumerate(self.train_data):
            x, y = x.to(self.device), y.to(self.device)

            # Forward Pass - prediction and its error
            pred = self.model(x)
            loss = self.loss_fn(pred, y)

            # Backward Pass - update parameters (weights, bias)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_train.append(loss)

    def evaluate(self):
        y_pred = []
        y_true = []

        self.loss_validate = []
        self.model.eval()

        with torch.no_grad():
            for batch, (x, y) in enumerate(self.test_data):
                x, y = x.to(self.device), y.to(self.device)

                # Forward Pass
                pred = self.model(x)
                loss = self.loss_fn(pred, y)

                # Save batch loss
                self.loss_validate.append(loss)

                # Save predictions and expected values
                y_pred.extend(pred)
                y_true.extend(y)

        return Metrics(y_true, y_pred)
    
    def mean_loss(self):
        loss_t = torch.mean(torch.FloatTensor(self.loss_train))
        loss_v = torch.mean(torch.FloatTensor(self.loss_validate))
        return loss_t, loss_v


# %%
#config.activation_fn = "sigomid"
#mlp = MultiLayerPerceptron()
#trainer = Trainer(config, mlp)


