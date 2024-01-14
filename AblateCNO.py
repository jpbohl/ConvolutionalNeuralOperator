import copy
import json
import os
import sys
import atexit

import pandas as pd
import torch
from tqdm import tqdm

import wandb
import matplotlib.pyplot as plt

from Problems.Straka import Straka
from TrainUtils import Trainer

if len(sys.argv) == 1:

    cluster = False
    mode = "disabled"

    training_properties = {
        "learning_rate": 0.001, 
        "weight_decay": 1e-6,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "epochs": 2,
        "batch_size": 16,
        "exp": 1, #Do we use L1 or L2 errors? Default: L1
        "training_samples": 4, #How many training samples?
    }
    
    with open("cno_architecture.json", "r") as f:
        model_architecture_ = json.loads(f.read().replace("\'", "\""))

    ablation = pd.read_csv("Ablation.csv") 
    counter = 22
    
    time = 300
    dataloc = "/Users/jan/sempaper/StrakaData/"

    # Save the models here:
    folder = "TrainedModels/"
        
else:
    
    cluster = True
    mode = "online"

    # Do we use a script to run the code (for cluster):
    folder = sys.argv[1] 

    # Reading hyperparameters
    with open(sys.argv[2], "r") as f:
        training_properties = json.loads(f.read().replace("\'", "\""))
    with open(sys.argv[3], "r") as f:
        model_architecture_ = json.loads(f.read().replace("\'", "\""))
    ablation = pd.read_csv(sys.argv[4])

    # Determine problem to run and data location
    time = int(sys.argv[5])
    dataloc = sys.argv[6]

    # Ablation study iteration 
    counter = int(os.environ["SLURM_ARRAY_TASK_ID"])

h
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Update model architecture with ablation parameters
ablation = dict(ablation.iloc[counter])
for name, val in ablation.items():
    model_architecture_[name] = val

wandb.login()
config = {"time" : time, **training_properties, **model_architecture_}
run = wandb.init(
    project = "StrakaCNO", 
    mode = mode,
    name = f"ablation-{counter}",
    config=config)
folder += run.name

#-----------------------------------Train--------------------------------------

# Get fixed training properties
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
training_samples = training_properties["training_samples"]
p = training_properties["exp"]
s = model_architecture_["in_size"]

if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

example = Straka(model_architecture_, device, batch_size, training_samples, time=time, s=s, dataloc=dataloc, cluster=cluster)
trainer = Trainer(training_properties, example, device)

# Training loop
print("Training")
for epoch in range(epochs):

    train_mse = trainer.train_epoch()
    val_loss = trainer.validate()

    print(f"Epoch {epoch} // MSE {train_mse} // Relative Val Loss {val_loss}")
        
    wandb.log(({
        "Train Loss" : train_mse,
        "Relative L2 Test Error" : val_loss}), 
        step=epoch)

