import copy
import json
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm

import wandb
import matplotlib.pyplot as plt

from Problems.Straka import StrakaFNO
from datetime import date

if len(sys.argv) == 1:

    cluster = False

    training_properties = {
        "learning_rate": 0.001,
        "weight_decay": 1e-8,
        "scheduler_step": 0.97,
        "scheduler_gamma": 10,
        "epochs": 2,
        "batch_size": 16,
        "exp": 1,
        "training_samples": 3,
    }
    fno_architecture_ = {
        "width": 16,
        "modes": 16,
        "FourierF" : 0, #Number of Fourier Features in the input channels. Default is 0.
        "n_layers": 2, #Number of Fourier layers
        "retrain": 4, #Random seed
        "in_size": 128,
        "self_attention" : True
    }
    
    #   "which_example" can be 
    
    #   poisson             : Poisson equation 
    #   wave_0_5            : Wave equation
    #   cont_tran           : Smooth Transport
    #   disc_tran           : Discontinuous Transport
    #   allen               : Allen-Cahn equation
    #   shear_layer         : Navier-Stokes equations
    #   airfoil             : Compressible Euler equations
    

    which_example = "straka"
    time = 600

    # Save the models here:
    folder = "TrainedModels/"
    dataloc = "/Users/jan/sempaper/StrakaData/"

else:
    # Do we use a script to run the code (for cluster):
    cluster = True
    folder = sys.argv[1]
    
    # Reading hyperparameters
    with open(sys.argv[2], "r") as f:
        training_properties = json.loads(f.read().replace("\'", "\""))
    with open(sys.argv[3], "r") as f:
        fno_architecture_ = json.loads(f.read().replace("\'", "\""))

    # Determine problem to run and data location
    which_example = sys.argv[4]
    time = int(sys.argv[5])
    dataloc = sys.argv[6]
    print("Loaded arguments")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initializing wandb
config = {"time" : time, **training_properties, **fno_architecture_}
wandb.login()
run = wandb.init(
    project = "StrakaFNO", 
    config=config)
folder += run.name # Change save directory name to wandb run name

learning_rate = training_properties["learning_rate"]
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
weight_decay = training_properties["weight_decay"]
scheduler_step = training_properties["scheduler_step"]
scheduler_gamma = training_properties["scheduler_gamma"]
training_samples = training_properties["training_samples"]
p = training_properties["exp"]
s = fno_architecture_["in_size"]


if which_example == "straka":
    print("Loading example")
    example = StrakaFNO(fno_architecture_, device, batch_size, training_samples,time=time, s=s, dataloc=dataloc, cluster=cluster)
    print("Loaded example")

else:
    raise ValueError("Problem not implemented")

if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([fno_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')

model = example.model
n_params = model.print_size()
train_loader = example.train_loader
test_loader = example.val_loader

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

# Initializing training parameters
freq_print = 1
if p == 1:
    loss = torch.nn.SmoothL1Loss()
elif p == 2:
    loss = torch.nn.MSELoss()
best_model_testing_error = 300
patience = int(0.1 * epochs)
counter = 0

def log_plots(model, val_loader):
    """
    Plotting function called when training loop exits.
    Plots the predictions on a batch of the test set and
    uploads them to wandb.
    """
    # Get single test batch
    input_batch, output_batch = next(iter(val_loader))
    input_batch = input_batch.to(device)
    output_batch = output_batch.to(device)

    with torch.no_grad():
        pred = model(input_batch)
        diffs = output_batch - pred

    # Logging initial conidition channel of inputs as well as outputs and
    # differences between predictions and labels
    input_img = input_batch[0, :, :, 2].cpu().numpy().T
    pred_img = pred[0, :, :, 0].cpu().numpy().T
    label = output_batch[0, :, :, 0].cpu().numpy().T
    diff_img = diffs[0, :, :, 0].cpu().numpy().T

    # Plotting intial condition
    figi, ax = plt.subplots()

    ic = ax.pcolormesh(input_img, cmap="Blues_r")
    ax.set_title("Initial condition")
    figi.colorbar(ic, ax=ax)

    # Plotting predictions
    figp, axes = plt.subplots(1, 2, sharey=True)

    labels = axes[0].pcolormesh(label, cmap="Blues_r")
    axes[0].set_title("Labels")
    figp.colorbar(labels, ax=axes[0])

    fno = axes[1].pcolormesh(pred_img, cmap="Blues_r")
    axes[1].set_title("FNO")
    figp.colorbar(fno, ax=axes[1])

    # Plotting errors
    fige, axes = plt.subplots(1, 2, sharey=True)

    labels = axes[0].pcolormesh(label, cmap="Blues_r")
    axes[0].set_title("Labels")
    fige.colorbar(labels, ax=axes[0])

    error_fno = axes[1].pcolormesh(diff_img, cmap="Blues_r")
    axes[1].set_title("FNO Error")
    fige.colorbar(error_fno, ax=axes[1])

    wandb.log({"Initial conditions" : wandb.Image(figi)})
    wandb.log({"Predictions" : wandb.Image(figp)})
    wandb.log({"Differences" : wandb.Image(fige)})

    print("Exiting code")

# Training loop
for epoch in range(epochs):
    with tqdm(unit="batch", disable=False) as tepoch:
        model.train()
        tepoch.set_description(f"Epoch {epoch}")
        train_mse = 0.0
        running_relative_train_mse = 0.0
        for step, (input_batch, output_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)
            output_pred_batch = model(input_batch)

            loss_f = loss(output_pred_batch, output_batch) / loss(torch.zeros_like(output_batch).to(device), output_batch)

            loss_f.backward()
            optimizer.step()
            train_mse = train_mse * step / (step + 1) + loss_f.item() / (step + 1)
            tepoch.set_postfix({'Batch': step + 1, 'Train loss (in progress)': train_mse})
                        
        wandb.log(({"Train Loss" : train_mse}), step=epoch)

        with torch.no_grad():
            model.eval()
            test_relative_l2 = 0.0
            train_relative_l2 = 0.0

            for step, (input_batch, output_batch) in enumerate(test_loader):
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                output_pred_batch = model(input_batch)
                
                loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(test_loader)
            
            wandb.log(({"Relative L2 Test Error" : test_relative_l2}), step=epoch)

            if test_relative_l2 < best_model_testing_error:
                best_model_testing_error = test_relative_l2
                best_model = copy.deepcopy(model)
                torch.save(best_model, folder + "/model.pkl")
                wandb.log(({"Best Relative Test Error" : best_model_testing_error}), step=epoch)
                counter = 0
                
            else:
                counter +=1

        tepoch.set_postfix({'Train loss': train_mse, "Relative Train": train_relative_l2, "Relative Val loss": test_relative_l2})
        tepoch.close()
        
        print(epoch, "val_loss/val_loss", test_relative_l2, epoch)
        
        with open(folder + '/errors.txt', 'w') as file:
            file.write("Training Error: " + str(train_mse) + "\n")
            file.write("Best Testing Error: " + str(best_model_testing_error) + "\n")
            file.write("Current Epoch: " + str(epoch) + "\n")
            file.write("Params: " + str(n_params) + "\n")
        scheduler.step()
    
    if counter > patience:
        print("Early Stopping")
        
log_plots(best_model, test_loader)
