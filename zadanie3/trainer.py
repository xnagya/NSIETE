# %% [markdown]
# # __Model trainer__
# This file contains Trainer and Statistics classes used during training of NN models. All metrics are calculated using library _torchmetrics_. 

# %%
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC
import pandas as pd
import torch.nn as nn
import time
import os
import numpy as np

# %%
import net_config as cfg

# %%
class EssayDataset(Dataset):
    def __init__(self, file_essay, file_missing):
        essay_arr = np.load(file_essay)
        indexes_arr = np.load(file_missing, allow_pickle=True)

        assert (essay_arr.shape[0] == indexes_arr.shape[0]), f"Wrong dataset size, essey count = {essay_arr.shape[0]} indexes count = {indexes_arr.shape[0]}"
        essey_count = essay_arr.shape[0]

        self.inputs = torch.from_numpy(essay_arr).type(torch.long)

        # List of tensors with target words
        self.targets = []

        # List of tensors with missing words index
        self.missing_positions = []

        # Extract missing words and their positions from numpy
        for i in range(essey_count):
            missing_words = []
            missing_pos = []

            for pair in indexes_arr[i]:
                pos = pair[0]
                word_idx = pair[1]

                missing_pos.append(pos)
                missing_words.append(word_idx)

            self.missing_positions.append(torch.tensor(missing_pos, dtype=torch.long))
            self.targets.append(torch.tensor(missing_words, dtype=torch.long))

            # Set missing word count per essay
            if (i == 0): 
                self.missing_per_essay = len(missing_words)
            
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        pos = self.missing_positions[idx]
        y = self.targets[idx]

        return x, pos, y

# %%
class Statistics:
    def __init__(self):
        self.metrics = dict()

    def update(self, metric_name, new_value):
        if metric_name in self.metrics:
            values = self.metrics[metric_name]
            values.append(new_value)
        else:
            values = [new_value]
            self.metrics.update({metric_name : values})

    def get_metric(self, metric_name):
        return self.metrics.get(metric_name)
    
    def batch_count(self):
        max = 0
        for val in self.metrics.values():
            if len(val) > max:
                max = len(val)

        return max 
    
    def clear(self):
        self.metrics = dict()

    # First batch is 0
    def batch_metrics(self, batch_num):
        result = dict()
        
        for metric_name, values in self.metrics.items():
            if (batch_num >= 0) and (batch_num < len(values)):
                metric_val = values[batch_num]
                result.update({metric_name : metric_val})

        return result
    
    def metric_average(self, metric_name):
        if metric_name in self.metrics:
            values = self.metrics[metric_name]
            return float(sum(values) / len(values))
        
        else: 
            return None

# %%
class Trainer:
    def __init__(self, model: nn.Module, vocab_size):
        # Select GPU device
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        print(f"Using {self.device} device for training")

        # Move model to available device
        self.network = model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters()
            , lr = cfg.learning_rate
            , betas = cfg.betas
            , weight_decay = cfg.weight_decay
        )

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

        # Class for saving metrics
        self.stats = Statistics()

        # Metrics 
        self.acc = MulticlassAccuracy(
            num_classes = vocab_size
            , average = "macro"
            ).to(self.device)
        self.roc = MulticlassAUROC(
            num_classes = vocab_size
            , average = "macro"
            ).to(self.device)
        self.f1 = MulticlassF1Score(
            num_classes = vocab_size
            , average = "macro"
            ).to(self.device)

        # Saving and loading model
        self.best_model = None
        self.best_accuracy = None

    def load_dataset(self, essay_path, positions_path):
        # Load dataset
        dataset = EssayDataset(essay_path, positions_path)
        
        # Split dataset to train, validation and test 
        gen = torch.Generator().manual_seed(42)
        data_train, data_val, data_test = random_split(dataset, [0.7, 0.15, 0.15], generator=gen)

        # Create dataset loaders
        self.data_train = DataLoader(data_train, batch_size = cfg.batch_size, shuffle = True)
        self.data_val = DataLoader(data_val, batch_size = cfg.batch_size, shuffle = True)
        self.data_test = DataLoader(data_test, batch_size = cfg.batch_size, shuffle = False)

    def save_model(self, current_epoch):  
        if self.best_model is not None:
            checkpoint = {
                'epoch': current_epoch,
                'NNmodel': self.best_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(checkpoint, cfg.model_path)
            print(f"NN model saved at path '{cfg.model_path}'")

    def load_model(self):
        checkpoint = torch.load(cfg.model_path)

        self.network.load_state_dict(checkpoint['NNmodel'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.test_model()

        print(f"NN model loaded from path '{cfg.model_path}'")
        return checkpoint['epoch']

    def train_model(self):
        # Train model (dataset = data_train)
        self.network.train()
        start = time.time()

        for x, pos, y in self.data_train:
            x, pos, y = x.to(self.device), pos.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            x = self.network(x, pos)

            # Calculate loss
            loss = self.loss_fn(x.view(-1, x.shape[-1]), y.view(-1))

            # Save batch loss
            self.stats.update("loss_train", loss)

            # Backward pass
            loss.backward()

            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), cfg.grad_clip)
            self.stats.update("grad", grad_norm)

            self.optimizer.step()

        end = time.time()
        print(f"Train time in sec = {end - start}")

        # Evaulate model by calculating loss (dataset = data_val)
        self.network.eval()
        start = time.time()

        with torch.no_grad():
            for x, pos, y in self.data_val:
                x, pos, y = x.to(self.device), pos.to(self.device), y.to(self.device)

                # Forward pass
                x = self.network(x, pos)

                # Calculate loss
                loss = self.loss_fn(x.view(-1, x.shape[-1]), y.view(-1))

                # Save batch loss
                self.stats.update("loss_val", loss)

        end = time.time()
        print(f"Validation time in sec = {end - start}")

    # Test model by calculating metrics (dataset = data_test) and keep the best model
    def test_model(self):
        self.network.eval()
        start = time.time()

        with torch.no_grad():
            for x, pos, y in self.data_test:
                x, pos, y = x.to(self.device), pos.to(self.device), y.to(self.device)

                # Forward pass
                x = self.network(x, pos)

                classes = torch.argmax(x, dim=2)
                confidence = torch.softmax(x, dim=2)
                
                y = y.view(-1)
                classes = classes.view(-1)
                confidence = confidence.view(-1, confidence.shape[2])

                # Calculate metrics
                accuracy = self.acc(classes, y).item()
                self.stats.update("acc", accuracy)
                self.stats.update("f1", self.f1(classes, y).item())
                # Calculating auroc takes too long - about 8x increase for test time       
                #self.stats.update("auroc", self.roc(confidence, y).item())
                
        end = time.time()
        print(f"Test time in sec = {end - start}")

        # Save best model
        if (self.best_accuracy is None) or (self.best_accuracy < accuracy):
            self.best_accuracy = accuracy
            self.best_model = self.network

# %%
"""
import models

embedding_path = "embedding_matrix.npy"

net = models.RNN("lstm", embedding_path, cfg.config_to_dict(cfg.config_NN))
t = Trainer(net, cfg.config_NN.vocab_size)

path_essay = "C:\\Users\\matul\\Desktop\\NSIETE\\zadanie3\\output\\essays_tensor_representation_max50_1miss.npy"
path_pos = "C:\\Users\\matul\\Desktop\\NSIETE\\zadanie3\\output\\position_index_pairs_max50_1miss.npy"

t.load_dataset(path_essay, path_pos)
t.train_model()
t.test_model()

data = EssayDataset(path_essay, path_pos)
data.__getitem__(50)
"""


