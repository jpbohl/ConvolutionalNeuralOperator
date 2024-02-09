import copy
import json
import os
import sys
import wandb

import pandas as pd
import torch

import matplotlib.pyplot as plt

from Problems.Straka import StrakaFNO
from Problems.StrakaMB import StrakaFNO as StrakaFNOMB
from TrainUtils import Trainer
from datetime import date

if len(sys.argv) == 1:

    cluster = False
    mode = "disabled"

    training_properties = {
        "learning_rate": 0.001,
        "weight_decay": 1e-8,
        "scheduler_step": 0.97,
        "scheduler_gamma": 10,
        "epochs": 10,
        "batch_size": 16,
        "exp": 1,
        "training_samples": 3,
        "loss" : "weighted"
    }
    fno_architecture_ = {
        "width": 16,
        "modes": 16,
        "FourierF" : 0, #Number of Fourier Features in the input channels. Default is 0.
        "n_layers": 2, #Number of Fourier layers
        "retrain": 4, #Random seed
        "in_size": 256,
        "attention" : False,
        "key_dim" : 8,
        "value_dim" : 16,
    }
    
    which_example = "StrakaMB"
    time = 900

    # Save the models here:
    folder = "TrainedModels/"
    dataloc = "/Users/jan/sempaper/StrakaMB/data/"

else:
    # Do we use a script to run the code (for cluster):
    cluster = True
    mode = "online"
    folder = sys.argv[1]
    
    # Reading hyperparameters
    with open(sys.argv[2], "r") as f:
        training_properties = json.loads(f.read().replace("\'", "\""))
    with open(sys.argv[3], "r") as f:
        fno_architecture_ = json.loads(f.read().replace("\'", "\""))

    # Determine problem to run and data location
    time = int(sys.argv[4])
    dataloc = sys.argv[5]
    which_example = sys.argv[6]
    print("Loaded arguments")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initializing wandb
config = {"time" : time, **training_properties, **fno_architecture_}
wandb.login()
run = wandb.init(
    project = which_example + "FNO",
    mode = mode,
    config=config)
folder += (run.name + "/") # Change save directory name to wandb run name

learning_rate = training_properties["learning_rate"]
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
weight_decay = training_properties["weight_decay"]
scheduler_step = training_properties["scheduler_step"]
scheduler_gamma = training_properties["scheduler_gamma"]
training_samples = training_properties["training_samples"]
p = training_properties["exp"]
s = fno_architecture_["in_size"]

torch.use_deterministic_algorithms(True, warn_only=True)

if which_example == "Straka":
    example = StrakaFNO(fno_architecture_, device, batch_size, training_samples, time=time, s=s, dataloc=dataloc, cluster=cluster)
elif which_example == "StrakaMB":
    example = StrakaFNOMB(fno_architecture_, device, batch_size, training_samples, time=time, s=s, dataloc=dataloc, cluster=cluster)
else:
    raise ValueError("Problem not implemented")

if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

with open(folder + "training_properties.json", "w") as f:
    f.write(json.dumps(training_properties))

with open(folder + "net_architecture.json", "w") as f:
    f.write(json.dumps(fno_architecture_))
    
model = example.model
n_params = model.print_size()
train_loader = example.train_loader
test_loader = example.val_loader

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

# Initializing training parameters
loss = torch.nn.MSELoss()
best_model_testing_error = 1000
patience = int(0.1 * epochs)
counter = 0

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

    if True:
        
        # Saving model
        weights = trainer.model.state_dict()
        torch.save(weights, folder + "/model_weights.pt")
        torch.save(trainer.model, folder + "/model.pkl")

        best_model_testing_error = val_loss
        counter = 0

    if counter > patience:
        print("Early stopping")
        break

    counter += 1