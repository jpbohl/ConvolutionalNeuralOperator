import copy
import json
import os
import sys

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Problems.Benchmarks import Airfoil, DiscContTranslation, ContTranslation, AllenCahn, SinFrequency, WaveEquation, ShearLayer
from Problems.Straka import Straka

if len(sys.argv) == 1:

    training_properties = {
        "learning_rate": 0.0005, 
        "weight_decay": 1e-10,
        "scheduler_step": 10,
        "scheduler_gamma": 0.98,
        "epochs": 2,
        "batch_size": 16,
        "exp": 1, #Do we use L1 or L2 errors? Default: L1
        "training_samples": 4, #How many training samples?
    }
    model_architecture_ = {
       
        #----------------------------------------------------------------------
        #Parameters to be chosen with model selection:
            
        "N_layers": 2, #Number of (D) + (U) layers. In our experiments, N_layers must be even.
        "kernel_size": 3, #Kernel size.
        "channel_multiplier": 32, #Parameter d_e (how the number of channels changes)
        
        "N_res": 6, #Number of (R) blocks.
        "res_len": 2, #Coefficienr r in (R) definition.
        
        #----------------------------------------------------------------------
        #Parameters that depend on the problem: 
        
        "in_size": 256, #Resolution of the computational grid
        "retrain": 4, #Random seed
        
        #----------------------------------------------------------------------
        #We fix the following parameters:
        
        #Filter properties:
        "cutoff_den": 2.0001, #
        "lrelu_upsampling": 2, #Coefficient N_{\sigma}. Default is 2.
        "half_width_mult": 1, #Coefficient c_h. Default is 1
        "filter_size": 6, # 2xfilter_size is the number of taps N_{tap}. Default is 6.
        "radial_filter": 0, #Is the filter radially symmetric? Default is 0 - NO.

        "FourierF": 0, #Number of Fourier Features in the input channels. Default is 0.

        #----------------------------------------------------------------------
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

    dataloc = "/Users/jan/sempaper/straka_data/"

    # Save the models here:
    folder = "TrainedModels/"+"CNO_"+which_example
        
else:
    
    # Do we use a script to run the code (for cluster):
    folder = sys.argv[1] + f"CNOStraka{sys.argv[5]}"
    
    # Reading hyperparameters
    with open(sys.argv[2], "r") as f:
        training_properties = json.loads(f.read().replace("\'", "\""))
    with open(sys.argv[3], "r") as f:
        model_architecture_ = json.loads(f.read().replace("\'", "\""))

    # Determine problem to run and data location
    which_example = sys.argv[4]
    time = sys.argv[5]
    dataloc = sys.argv[6]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=folder) #usage of TensorBoard

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

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([model_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')

if which_example == "shear_layer":
    example = ShearLayer(model_architecture_, device, batch_size, training_samples)
elif which_example == "poisson":
    example = SinFrequency(model_architecture_, device, batch_size, training_samples, dataloc=dataloc)
elif which_example == "wave_0_5":
    example = WaveEquation(model_architecture_, device, batch_size, training_samples, dataloc=dataloc)
elif which_example == "allen":
    example = AllenCahn(model_architecture_, device, batch_size, training_samples)
elif which_example == "cont_tran":
    example = ContTranslation(model_architecture_, device, batch_size, training_samples)
elif which_example == "disc_tran":
    example = DiscContTranslation(model_architecture_, device, batch_size, training_samples)
elif which_example == "airfoil":
    example = Airfoil(model_architecture_, device, batch_size, training_samples)
elif which_example == "straka":
    example = Straka(model_architecture_, device, batch_size, training_samples, time=time, s=s, dataloc=dataloc)
else:
    raise ValueError()

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
patience = int(0.1 * epochs)    # Earlt stopping parameter
counter = 0

for epoch in range(epochs):
    with tqdm(unit="batch", disable=False) as tepoch:
        
        #Disable : Should we disable the printing of the error report per epoch?
        
        model.train()
        tepoch.set_description(f"Epoch {epoch}")
        train_mse = 0.0
        running_relative_train_mse = 0.0
        for step, (input_batch, output_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)

            output_pred_batch = model(input_batch)

            if which_example == "airfoil": #Mask the airfoil shape
                output_pred_batch[input_batch==1] = 1
                output_batch[input_batch==1] = 1

            loss_f = loss(output_pred_batch, output_batch) / loss(torch.zeros_like(output_batch).to(device), output_batch)

            loss_f.backward()
            optimizer.step()
            train_mse = train_mse * step / (step + 1) + loss_f.item() / (step + 1)
            tepoch.set_postfix({'Batch': step + 1, 'Train loss (in progress)': train_mse})

        writer.add_scalar("train_loss/train_loss", train_mse, epoch)

        with torch.no_grad():
            model.eval()
            test_relative_l2 = 0.0
            train_relative_l2 = 0.0
            
            for step, (input_batch, output_batch) in enumerate(val_loader):
                
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                output_pred_batch = model(input_batch)
                
                if which_example == "airfoil": #Mask the airfoil shape
                    output_pred_batch[input_batch==1] = 1
                    output_batch[input_batch==1] = 1
                
                loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(val_loader)

            for step, (input_batch, output_batch) in enumerate(train_loader):
                    input_batch = input_batch.to(device)
                    output_batch = output_batch.to(device)
                    output_pred_batch = model(input_batch)
                    
                    if which_example == "airfoil": #Mask the airfoil shape
                        output_pred_batch[input_batch==1] = 1
                        output_batch[input_batch==1] = 1

                    loss_f = torch.mean(abs(output_pred_batch - output_batch)) / torch.mean(abs(output_batch)) * 100
                    train_relative_l2 += loss_f.item()
            train_relative_l2 /= len(train_loader)

            writer.add_scalar("train_loss/train_loss_rel", train_relative_l2, epoch)
            writer.add_scalar("val_loss/val_loss", test_relative_l2, epoch)

            if test_relative_l2 < best_model_testing_error:
                best_model_testing_error = test_relative_l2
                best_model = copy.deepcopy(model)
                torch.save(best_model, folder + "/model.pkl")
                writer.add_scalar("val_loss/Best Relative Testing Error", best_model_testing_error, epoch)
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