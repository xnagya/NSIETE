# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import Dice, MulticlassAccuracy
import time

import net_config as cfg

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
def pixel_accuracy(prediction: torch.Tensor, truth: torch.Tensor):
    with torch.no_grad():
        pixel_count = float(truth.numel())
        correct = (torch.eq(prediction, truth).int()).sum()

        accuracy = float(correct) / pixel_count

    return accuracy


# %%
class Trainer:
    def __init__(self, model: nn.Module):
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
            , lr=cfg.learning_rate
            , betas=cfg.betas
            , weight_decay=cfg.weight_decay
        )

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=cfg.background_class)

        # Statistics (metrics for each epoch)
        self.stats = Statistics()

        # Metrics
        self.acc = MulticlassAccuracy(ignore_index=cfg.background_class, num_classes=cfg.num_of_classes, average='micro')
        self.dice = Dice(ignore_index=cfg.background_class, num_classes=cfg.num_of_classes, average='micro')

        # Saving and loading model
        self.best_model = None
        self.best_accuracy = None

    # Create Data Loaders
    def load_dataset(self, train_data, val_data, test_data):
        self.train_data = DataLoader(train_data, batch_size= cfg.batch_size, shuffle= True)
        self.val_data = DataLoader(val_data, batch_size= cfg.batch_size, shuffle= True)
        self.test_data = DataLoader(test_data, batch_size= cfg.batch_size, shuffle= False)

    def save_model(self, current_epoch):  
        if self.best_model is not None:
            checkpoint = {
                'epoch': current_epoch,
                'model_state_dict': self.best_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            torch.save(checkpoint, cfg.model_path)
            print(f"NN model saved at path '{cfg.model_path}'")

    def load_model(self):
        checkpoint = torch.load(cfg.model_path)

        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.evaluate_model()

        print(f"NN model loaded from path '{cfg.model_path}'")
        return checkpoint['epoch']
    
    # Forward Pass - create prediction and return its error (loss)
    def forward_pass(self, input, ground_truth):
        prediction = self.network(input)
        loss = self.loss_fn(prediction, ground_truth)
        return loss
    
    # Backward Pass - update parameters (weights, bias)
    def backward_pass(self, loss_value):
        self.optimizer.zero_grad()
        loss_value.backward()
        self.optimizer.step()

    def train_model(self):
        # Train model (dataset = train_data)
        self.network.train()
        start = time.time()

        for x, y in self.train_data:
            x, y = x.to(self.device), y.to(self.device)

            loss = self.forward_pass(x, y)

            # Save batch loss
            self.stats.update(cfg.metric_name_Tloss, loss)

            self.backward_pass(loss)

        end = time.time()
        print(f"Train time in sec = {end - start}")

        # Evaulate model by calculating loss (dataset = val_data)
        self.network.eval()
        start = time.time()

        with torch.no_grad():
            for x, y in self.val_data:
                x, y = x.to(self.device), y.to(self.device)

                loss = self.forward_pass(x, y)

                # Save batch loss
                self.stats.update(cfg.metric_name_Vloss, loss)

        end = time.time()
        print(f"Validation time in sec = {end - start}")

    # Evaulate model by calculating metrics (dataset = test_data) and keep best model
    def evaluate_model(self):
        self.network.eval()
        start = time.time()

        with torch.no_grad():
            for x, y in self.test_data:
                x, y = x.to(self.device), y.to(self.device)

                pred = self.network(x)
                classes = torch.argmax(pred, dim = 1)

                print(classes.shape)
                print(torch.unique(classes))

                # Calculate metrics
                a = self.acc(classes, y).item()
                d = self.dice(classes, y).item()

                self.stats.update(cfg.metric_name_acc, a)
                self.stats.update(cfg.metric_name_dice, d)

        end = time.time()
        print(f"Test time in sec = {end - start}")

        # Get best model
        if (self.best_accuracy is None) or (self.best_accuracy < a):
            self.best_accuracy = a
            self.best_model = self.network()



