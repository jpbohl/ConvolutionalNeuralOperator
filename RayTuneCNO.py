import copy
import json
import os
import tempfile
import sys

import pandas as pd
import torch
from tqdm import tqdm

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
import matplotlib.pyplot as plt

from Problems.Straka import Straka

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
        "in_size": 128,            # Resolution of the computational grid
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

    # Determine problem to run and data location
    time = int(sys.argv[4])
    dataloc = sys.argv[5]

#-----------------------------------Train--------------------------------------
    
class Trainer():

    def __init__(self, training_properties, example, device):
        
        # Loading training properties
        learning_rate = training_properties["learning_rate"]
        weight_decay = training_properties["weight_decay"]
        scheduler_step = training_properties["scheduler_step"]
        scheduler_gamma = training_properties["scheduler_gamma"]

        # Get model and dataset
        self.model = example.model
        n_params = self.model.print_size()
        self.train_loader = example.train_loader #TRAIN LOADER
        self.val_loader = example.val_loader #VALIDATION LOADER
        
        # Set up optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        
        self.loss = torch.nn.MSELoss()
        self.device = device

    def train_epoch(self, epoch):

        self.model.train()
        train_mse = 0.0
        running_relative_train_mse = 0.0
        for step, (input_batch, output_batch) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            input_batch = input_batch.to(self.device)
            output_batch = output_batch.to(self.device)

            output_pred_batch = self.model(input_batch)

            loss_f = self.loss(output_pred_batch, output_batch) / self.loss(torch.zeros_like(output_batch).to(self.device), output_batch)

            loss_f.backward()
            self.optimizer.step()
            train_mse = train_mse * step / (step + 1) + loss_f.item() / (step + 1)

        self.scheduler.step()


        return train_mse

    def validate(self):
        
        with torch.no_grad():
            self.model.eval()
            test_relative_l2 = 0.0
            train_relative_l2 = 0.0
            
            for step, (input_batch, output_batch) in enumerate(self.val_loader):
                
                input_batch = input_batch.to(self.device)
                output_batch = output_batch.to(self.device)
                output_pred_batch = self.model(input_batch)
                
                loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(self.val_loader)

            return test_relative_l2

# Training loop
def train_model(config):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Modify model parameters given by hyperparameter search
    model_architecture_["N_res"] = config["N_res"]
    model_architecture_["N_res_neck"] = config["N_res_neck"]
    model_architecture_["channel_multiplier"] = config["channel_multiplier"]

    # Get fixed training properties
    epochs = training_properties["epochs"]
    batch_size = training_properties["batch_size"]
    training_samples = training_properties["training_samples"]
    p = training_properties["exp"]
    s = model_architecture_["in_size"]

    if not os.path.isdir(folder):
        print("Generated new folder")
        os.mkdir(folder)

    # Load dataset
    example = Straka(model_architecture_, device, batch_size, training_samples, time=time, s=s, dataloc=dataloc, cluster=cluster)

    # Setting up training agent
    trainer = Trainer(training_properties, example, device)

    # Training loop
    for epoch in range(epochs):

        train_mse = trainer.train_epoch(epoch)
        val_loss = trainer.validate()

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint.pt")
            torch.save((trainer.model.state_dict(), trainer.optimizer.state_dict()), path)
            
        tune.report(loss= val_loss)
        
        print(f"Epoch {epoch} // Train MSE {train_mse} // Test Loss {val_loss}")

def main(num_samples=10, max_num_epochs=600, gpus_per_trial=1):

    config = {
        "N_res": tune.grid_search([2, 4, 6]),
        "N_res_neck" : tune.grid_search([4, 6, 8]),
        "channel_multiplier" : tune.grid_search([12, 16, 32])
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    results = tune.run(
        tune.with_parameters(train_model),
        resources_per_trial = {"cpu": 8, "gpu": gpus_per_trial},
        config = config,
        metric="loss",
        mode="min",
        scheduler=scheduler,
        num_samples=num_samples)

    best_trial = result.get_best_trial("loss", "min", "last")

    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.metrics["loss"]))

main(num_samples=1, max_num_epochs=20, gpus_per_trial=0)
