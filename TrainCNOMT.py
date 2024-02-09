import copy
import json
import os
import sys

import torch
import wandb

from TrainUtils import Trainer
from Problems.StrakaMT import Straka
from Problems.StrakaMTMB import Straka as StrakaMB

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
        "channel_multiplier": 16, # Parameter d_e (how the number of channels changes)
        "N_res": 2,               # Number of (R) blocks in the middle networs.
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
    project = "StrakaCNOMB", 
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

if str(device) == 'cpu':
    print("------------------------------------------")
    print("YOU ARE RUNNING THE CODE ON A CPU.")
    print("WE SUGGEST YOU TO RUN THE CODE ON A GPU!")
    print("------------------------------------------")
    print(" ")


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
