# %% [markdown]
# # __WanDB Experiment__
# This file connects _models.py_ and _trainer.py_ files and manages experiments created in wanDB. Experiments are defined in NN_z3.ipynb file.

# %%
import wandb
import torch.nn as nn
import gc
import os.path

# %%
import net_config as cfg
from models import *
from trainer import *

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
        
        #wandb.finish()
        
        self.run = wandb.init(
        entity = cfg.project_entity, 
        project = cfg.project_name,     
        name = run_name, 
        id = run_id, 
        reinit=True, 
        #settings = wandb.Settings(start_method="fork")
        )

        wandb.config = cfg.config_to_dict(cfg.config_NN)

        self.trainer = Trainer(model, cfg.config_NN.vocab_size)
        self.save_interval = save_interval
        self.datasets_loaded = False
        self.batch_count = 0

        # Load best model
        if (self.save_interval is not None) and os.path.isfile(cfg.model_path):
            self.current_epoch = self.trainer.load_model()
        else:
            self.current_epoch = 0

    def load_datasets(self, essay_path, positions_path):
        self.trainer.load_dataset(essay_path, positions_path)
        self.datasets_loaded = True
    
    def execute_training(self, epoch_count, log_batch = False):
        assert self.datasets_loaded, "Datasets are NOT loaded"

        for _ in range(epoch_count):
            self.current_epoch += 1
            print(f"--Starting epoch {self.current_epoch}--")

            # Train model
            self.trainer.train_model()
            # Test model
            self.trainer.test_model()
            
            # Log results
            if log_batch:
                for i in range(self.trainer.stats.batch_count()):
                    self.batch_count += 1
                    batch_metrics = self.trainer.stats.batch_metrics(i)

                    self.run.log({"loss_train": batch_metrics.get("loss_train"), "batch": self.batch_count})
                    self.run.log({"loss_val": batch_metrics.get("loss_val"), "batch": self.batch_count})
                    self.run.log({"accuracy": batch_metrics.get("acc"), "batch": self.batch_count})
                    self.run.log({"f1_score": batch_metrics.get("f1"), "batch": self.batch_count})
                    #self.run.log({"auroc": batch_metrics.get("auroc"), "batch": self.batch_count})
                    self.run.log({"grad": batch_metrics.get("grad"), "batch": self.batch_count})
            else:
                # Get metrics average
                tl = self.trainer.stats.metric_average("loss_train")
                vl = self.trainer.stats.metric_average("loss_val")
                acc = self.trainer.stats.metric_average("acc")
                f1 = self.trainer.stats.metric_average("f1")
                #roc = self.trainer.stats.metric_average("auroc")
                g = self.trainer.stats.metric_average("grad")

                # Save metrics to wandb
                self.run.log({"loss_train": tl, "epoch": self.current_epoch})
                self.run.log({"loss_val": vl, "epoch": self.current_epoch})
                self.run.log({"accuracy": acc, "epoch": self.current_epoch})
                self.run.log({"f1_score": f1, "epoch": self.current_epoch})
                #self.run.log({"auroc": roc, "epoch": self.current_epoch})
                self.run.log({"grad": g, "epoch": self.current_epoch})

            self.trainer.stats.clear()
            gc.collect()

            # Save best model
            if (self.save_interval is not None) and (self.current_epoch % self.save_interval == 0):
                self.trainer.save_model(self.current_epoch)

            print(f"--Ending epoch {self.current_epoch}--")
    
    def stop_run(self):
        self.run.finish()
        del self.trainer
        self.datasets_loaded = False

        gc.collect()



