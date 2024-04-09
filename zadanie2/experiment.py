# %% [markdown]
# # __WanDB Experiment__
# This file connects _models.py_ and _trainer.py_ files and manages experiments created in wanDB. It also contains dataset reresentation as Dataset subclass (Lizard_dataset). Experiments are defined in file NN-z2 (main file).
# 

# %%
import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import gc
import os.path

# %%
import net_config as cfg
from models import *
from trainer import *

# %% [markdown]
# ## Dataset class
# This class is parsed to DataLoader in trainer.py 
# 
# Images are loaded to tensors __(dtype = float32)__ from argument 'path_images' </br>
# Labels are loaded to tensors __(dtype = int64)__ from argument 'path_labels'

# %%
class Lizard_dataset(Dataset):
    def __init__(self, path_images, path_labels):
        self.images = []
        self.labels = []
        self.transform = T.ToPILImage()

        self.class_colors = torch.FloatTensor([
            [0, 0, 0]           #black
            , [30, 144, 255]    #dodger blue
            , [220, 20, 60]     #crimson
            , [34, 139, 34]     #forest green
            , [238, 130, 238]   #violet
            , [255, 255, 0]     #yellow
            , [211, 211, 211]   #gainsboro
            ]
        )

        # Load images and labels as tensors to cpu (moved to gpu during training)
        self.images = torch.load(path_images, map_location="cpu").type(torch.float32)
        self.labels = torch.load(path_labels, map_location="cpu").type(torch.int64)

        img_count = self.images.size(dim=0)
        num_channels = self.images.size(dim=1)
        height = self.images.size(dim=2)
        width = self.images.size(dim=3)

        img_count2 = self.labels.size(dim=0)
        height2 = self.labels.size(dim=1)
        width2 = self.labels.size(dim=2)

        assert img_count == img_count2, f"Wrong IMAGE COUNT for dataset: images = {img_count} | labels = {img_count2}"
        assert height == height2, f"Wrong image HEIGHT for dataset: images = {height} | labels = {height2}"
        assert width == width2, f"Wrong image WIDTH for dataset: images = {width} | labels = {width2}"
        assert num_channels == 3, f"Wrong image CHANNEL COUNT for dataset: images = {num_channels} | needed = 3"
        
    def __len__(self):
        return self.images.size(dim=0)

    def __getitem__(self, idx):
        image = self.images[idx, :, :, :]
        label = self.labels[idx, :, :]

        return (image, label)
    
    def show_imgLabel(self, idx):
        img_t, label_t = self.__getitem__(idx)

        # Show image
        image = self.transform(img_t)

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(f'Original Image  with index = {idx}')
        plt.axis('off')
        plt.show()

        # show segmentation
        converted_tensor = torch.nn.functional.embedding(label_t.type(torch.int64), self.class_colors).permute(2, 0, 1)
        colormap = self.transform(converted_tensor)

        # show segmentation
        plt.figure(figsize=(8, 8))
        plt.imshow(colormap)
        plt.title('Segmentation Heatmap')
        plt.axis('off')
        plt.show()

    def show_tensorImg(self, t):
        t = torch.squeeze(t)
        img = self.transform(t)
        plt.show(img)
        

# %% [markdown]
# ## wanDB run class
# 
# This class executes training epochs by calling trainer functions. It also logs metrics and decides when the model params are saved (locally).
# This class contains: 
# - Current wanDB run 
# - Trainer
# - Save interval (every n-th epoch)
# 

# %%
class wanDB_run: 
    def __init__(self, run_name, run_id, model: nn.Module, save_interval = None):
        wandb.login()
        
        wandb.finish()
        
        self.run = wandb.init(
        entity = cfg.project_entity, 
        project = cfg.project_name,     
        name = run_name, 
        id = run_id
        )

        wandb.config = cfg.config_to_dict(cfg.config_Unet)

        self.trainer = Trainer(model)
        self.save_interval = save_interval
        self.datasets_loaded = False

        # Load best model
        if (self.save_interval is not None) and os.path.isfile(cfg.model_path):
            self.current_epoch = self.trainer.load_model()
        else:
            self.current_epoch = 0

    def load_datasets(self, train_pathX, train_pathY, val_pathX, val_pathY, test_pathX, test_pathY):
        trainData = Lizard_dataset(train_pathX, train_pathY)
        valData = Lizard_dataset(val_pathX, val_pathY)
        testData = Lizard_dataset(test_pathX, test_pathY)

        self.trainer.load_dataset(trainData, valData, testData)
        self.datasets_loaded = True
    
    def execute_training(self, epoch_count):
        assert self.datasets_loaded, "Datasets are NOT loaded"

        for _ in range(epoch_count):
            self.current_epoch += 1
            print(f"--Starting epoch {self.current_epoch}--")

            # Train model
            self.trainer.train_model()
            # Evaluate model
            self.trainer.evaluate_model()

            # Get metrics
            tl = self.trainer.stats.metric_average(cfg.metric_name_Tloss)
            vl = self.trainer.stats.metric_average(cfg.metric_name_Vloss)
            acc = self.trainer.stats.metric_average(cfg.metric_name_acc)
            iou = self.trainer.stats.metric_average(cfg.metric_name_iou)
            dice = self.trainer.stats.metric_average(cfg.metric_name_dice)

            # Save metrics to wandb
            self.run.log({"loss_train": tl, "epoch": self.current_epoch})
            self.run.log({"loss_val": vl, "epoch": self.current_epoch})
            self.run.log({"accuracy": acc, "epoch": self.current_epoch})
            self.run.log({"iou": iou, "epoch": self.current_epoch})
            self.run.log({"dice": dice, "epoch": self.current_epoch})

            # Save best model
            if (self.save_interval is not None) and (self.current_epoch % self.save_interval == 0):
                self.trainer.save_model(self.current_epoch)

            print(f"--Ending epoch {self.current_epoch}--")
    
    def stop_run(self):
        self.run.finish()
        del self.trainer
        self.datasets_loaded = False

        gc.collect()


# %%
"""
d = cfg.config_to_dict(cfg.config_Unet)

net = U_Net(3, cfg.num_of_classes, d)
#output_shapes(net, 500, 500)

t = torch.rand(14, 3, 500, 500)
print(t.shape)
t = net(t)
print(t.shape)

classes = torch.argmax(t, dim = 1)
print(classes.shape)
"""


