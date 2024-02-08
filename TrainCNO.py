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
from Problems.StrakaMB import Straka as StrakaMB
from Problems.StrakaMT import Straka as StrakaMT

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
    model_architecture_ = {
        
        #Parameters to be chosen with model selection:
        "N_layers": 3,            # Number of (D) & (U) blocks 
        "channel_multiplier": 32, # Parameter d_e (how the number of channels changes)
        "N_res": 4,               # Number of (R) blocks in the middle networs.
        "N_res_neck" : 6,         # Number of (R) blocks in the BN
        
        #Other parameters:
        "in_size": 256,            # Resolution of the computational grid
        "retrain": 4,             # Random seed
        "kernel_size": 3,         # Kernel size.
        "FourierF": 0,            # Number of Fourier Features in the input channels. Default is 0.
        "activation": 'cno_lrelu',# cno_lrelu or lrelu
        "attention" : True,
        
        #Filter properties:
        "cutoff_den": 2.0001,     # Cutoff parameter.
        "lrelu_upsampling": 2,    # Coefficient N_{\sigma}. Default is 2.
        "half_width_mult": 0.8,   # Coefficient c_h. Default is 1
        "filter_size": 6,         # 2xfilter_size is the number of taps N_{tap}. Default is 6.
        "radial_filter": 0,       # Is the filter radially symmetric? Default is 0 - NO.
    }
    
    #   "which_example" can be 
    
    #   poisson             : Poisson equation 
    #   wave_0_5            : Wave equation
    #   cont_tran           : Smooth Transport
    #   disc_tran           : Discontinuous Transport
    #   allen               : Allen-Cahn equation
    #   shear_layer         : Navier-Stokes equations
    #   airfoil             : Compressible Euler equations
    

    which_example = "StrakaMB"
    time = 300

    dataloc = "/Users/jan/sempaper/StrakaMB/data/"

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
    time = int(sys.argv[4])
    dataloc = sys.argv[5]
    which_example = sys.argv[6]

    if len(sys.argv) == 8:
        time0 = int(sys.argv[-1])
    else:
        time0 = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {"time" : time, "initial time" : time0, **training_properties, **model_architecture_}
wandb.login()

run = wandb.init(
    project = which_example + "CNO", 
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

with open(folder + "/training_properties.json", "w") as f:
    f.write(json.dumps(training_properties))
with open(folder + "/net_architecture.json", "w") as f:
    f.write(json.dumps(model_architecture_))

print("Loading example")
if which_example == "Straka":
    example = Straka(model_architecture_, device, batch_size, training_samples, time=time, s=s, dataloc=dataloc, cluster=cluster)

elif which_example == "StrakaMB":
    example = StrakaMB(model_architecture_, device, batch_size, training_samples, time=time, time0=time0, s=s, dataloc=dataloc, cluster=cluster)

print("Loaded example")
    
#-----------------------------------Train--------------------------------------
model = example.model
n_params = model.print_size()
train_loader = example.train_loader #TRAIN LOADER
val_loader = example.val_loader #VALIDATION LOADER

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
freq_print = 1

if p == 1:
    loss = torch.nn.L1Loss()
elif p == 2:
    loss = torch.nn.MSELoss()
    
best_model_testing_error = 1000 #Save the model once it has less than 1000% relative L1 error
patience = int(0.2 * epochs)    # Early stopping parameter
counter = 0

if str(device) == 'cpu':
    print("------------------------------------------")
    print("YOU ARE RUNNING THE CODE ON A CPU.")
    print("WE SUGGEST YOU TO RUN THE CODE ON A GPU!")
    print("------------------------------------------")
    print(" ")

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
    input_img = input_batch[0, -1, :, :].detach().cpu().numpy().T
    pred_img = pred[0, 0, :, :].cpu().numpy().T
    label = output_batch[0, 0, :, :].detach().cpu().numpy().T
    diff_img = diffs[0, 0, :, :].cpu().numpy().T

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
    axes[1].set_title("CNO")
    figp.colorbar(fno, ax=axes[1])

    # Plotting errors
    fige, axes = plt.subplots(1, 2, sharey=True)

    labels = axes[0].pcolormesh(label, cmap="Blues_r")
    axes[0].set_title("Labels")
    fige.colorbar(labels, ax=axes[0])

    error_fno = axes[1].pcolormesh(diff_img, cmap="Blues_r")
    axes[1].set_title("CNO Error")
    fige.colorbar(error_fno, ax=axes[1])

    wandb.log({"Initial conditions" : wandb.Image(figi)})
    wandb.log({"Predictions" : wandb.Image(figp)})
    wandb.log({"Differences" : wandb.Image(fige)})


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
            
            for step, (input_batch, output_batch) in enumerate(val_loader):
                
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                output_pred_batch = model(input_batch)
                
                loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(val_loader)

            wandb.log(({"Relative L2 Test Error" : test_relative_l2}), step=epoch)

            if test_relative_l2 < best_model_testing_error:
                best_model_testing_error = test_relative_l2
                best_model = copy.deepcopy(model)
                wandb.log(({"Best Relative Test Error" : best_model_testing_error}), step=epoch)
                counter = 0

            else:
                counter+=1

        tepoch.set_postfix({'Train loss': train_mse, "Relative Train": train_relative_l2, "Relative Val loss": test_relative_l2})
        tepoch.close()

        with open(folder + '/errors.txt', 'w') as file:
            file.write("Training Error: " + str(train_mse) + "\n")
            file.write("Best Testing Error: " + str(best_model_testing_error) + "\n")
            file.write("Current Epoch: " + str(epoch) + "\n")
            file.write("Params: " + str(n_params) + "\n")
        scheduler.step()

    if counter>patience:
        print("Early Stopping")
        break

torch.save(best_model.state_dict(), folder + "/model_weights.pt")
torch.save(best_model, folder + "/model.pkl")
torch.save(best_model(torch.ones_like(input_batch)).detach(), folder + "/ones.pt")



log_plots(best_model, val_loader)
