import copy
import json
import os
import sys

import torch
import wandb

from TrainUtils import Trainer
from Problems.StrakaMT import StrakaFNO as Straka
from Problems.StrakaMTMB import StrakaFNO as StrakaMB

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
    }
    model_architecture_ = {
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
    
    dataloc = "/Users/jan/sempaper/StrakaMB/data/"
    problem = "StrakaMB"

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

    # Determine problem to run and data location
    dataloc = sys.argv[4]
    problem = sys.argv[5]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mb = True if problem == "StrakaMB" else False

config = {**training_properties, **model_architecture_, "mb":mb}
wandb.login()
run = wandb.init(
    project = "StrakaFNOMB",
    mode = mode,
    config=config)
folder += run.name

learning_rate = training_properties["learning_rate"]
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
weight_decay = training_properties["weight_decay"]
scheduler_step = training_properties["scheduler_step"]
scheduler_gamma = training_properties["scheduler_gamma"]
training_samples = training_properties["training_samples"]
p = training_properties["exp"]
s = model_architecture_["in_size"]

if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

with open(folder + "training_properties.txt", "w") as f:
    f.write(json.dumps(training_properties))
with open(folder + "net_architecture.txt", "w") as f:
    f.write(json.dumps(model_architecture_))

print("Loading example")
if problem == "Straka":
    example = Straka(model_architecture_, device, batch_size, training_samples, s=s, dataloc=dataloc, cluster=cluster)
elif problem == "StrakaMB":
    example = StrakaMB(model_architecture_, device, batch_size, training_samples, s=s, dataloc=dataloc, cluster=cluster)
print("Loaded example")
    
#-----------------------------------Train--------------------------------------
model = example.model
n_params = model.print_size()
train_loader = example.train_loader #TRAIN LOADER
val_loader = example.val_loader #VALIDATION LOADER

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
freq_print = 1

loss = torch.nn.MSELoss()
    
best_model_testing_error = 1000 #Save the model once it has less than 1000% relative L1 error
patience = int(0.2 * epochs)    # Early stopping parameter
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

    if val_loss < best_model_testing_error:
        
        # Saving model
        weights = trainer.model.state_dict()
        torch.save(weights, folder + "/model_weights.pt")
        torch.save(trainer.model, folder + "/model.pkl")

        best_model_testing_error = val_loss
        counter = 0

    if counter > patience:
        break

    counter += 1