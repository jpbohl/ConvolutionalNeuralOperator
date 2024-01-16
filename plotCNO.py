import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import json
import sys

from Problems.Straka import StrakaDataset, Straka
from CNOModule import CNO

# Getting arguments for the timestep at which model was trained and
# the names of the runs to plot
time = int(sys.argv[1])
run = sys.argv[2]
cno = True

runloc = f"/cluster/work/math/jabohl/runs/{run}/"

# Loading config files
with open("cno_architecture.json", "r") as f:
    cno_config = json.loads(f.read())

with open("cno_training_properties.json", "r") as f:
    train_config = json.loads(f.read())

# Loading Model and Data
device = "cpu"

dataloc = "/cluster/work/math/jabohl/StrakaData/data/"
dataset = StrakaDataset(dataloc, time=time, ntest=56, which="validation", cno=cno)

N_res = cno_config["N_res"]
N_res_neck = cno_config["N_res_neck"]
N_layers = cno_config["N_layers"]
kernel_size = cno_config["kernel_size"]
channel_multiplier = cno_config["channel_multiplier"]
cutoff_den = cno_config["cutoff_den"]
filter_size = cno_config["filter_size"]
radial = cno_config["radial_filter"]
half_width_mult = cno_config["half_width_mult"]
lrelu_upsampling = cno_config["lrelu_upsampling"]
s = 256

model = CNO(in_dim=3,  # Number of input channels.
                        in_size=s,
                        cutoff_den=cutoff_den,
                        N_layers=N_layers,
                        N_res=N_res,
                        N_res_neck=N_res_neck,
                        radial=radial,
                        filter_size=filter_size,
                        conv_kernel=kernel_size,
                        lrelu_upsampling = lrelu_upsampling,
                        half_width_mult = half_width_mult,
                        channel_multiplier = channel_multiplier,
                        attention = False
                        ).to(device)

weights = torch.load(runloc + "model_weights.pt", map_location=device)
model.load_state_dict(weights)

# Computing prediction and difference between labels and prediction
model.eval()

# Plotting Testset Labels, Predictions and Error
nrows = 4
ncols = 4
loader = iter(torch.utils.data.DataLoader(dataset, batch_size=4))
fig1, axs1 = plt.subplots(nrows, ncols, sharex = True, sharey=True, layout="tight")
fig2, axs2 = plt.subplots(nrows, ncols, sharex = True, sharey=True, layout="tight")
fig3, axs3 = plt.subplots(nrows, ncols, sharex = True, sharey=True, layout="tight")

fig1.subplots_adjust(hspace=.5)
fig2.subplots_adjust(hspace=.5)
fig3.subplots_adjust(hspace=.5)

ys, yhats, errors = [], [], []
for i in range(nrows):
    x, y = next(loader) # Get batch from test loader
    
    y = y.detach().cpu().numpy()
    yhat = model(x.to(device)).detach().cpu().numpy()
    error = y - yhat
    
    ys.append(y)
    yhats.append(yhat)
    errors.append(error)
    
v_min_ys = min(y.min() for y in ys)
v_max_ys = max(y.max() for y in ys)

v_min_yhats = min(y.min() for y in yhats)
v_max_yhats = max(y.max() for y in yhats)

v_min_errors = min(y.min() for y in errors)
v_max_errors = max(y.max() for y in errors)

for i in range(nrows):

    y = ys[i]
    yhat = yhats[i]
    error = errors[i]

    # Relative L1 Error
    l1_loss = np.mean(abs(yhat - y), axis=(1,2,3)) / np.mean(abs(y)) * 100

    for j in range(ncols):
        
        # Getting viscosity and density
        v = round(x[j, 0, 0, 0].item(), 1)
        d = round(x[j, 1, 0, 0].item(), 1)

        # Plotting
        norm = colors.CenteredNorm()
        labels = axs1[i, j].pcolormesh(y[j, 0, :, :].T, 
                                       norm=norm, 
                                       cmap="seismic")
        
        norm = colors.CenteredNorm()
        preds = axs2[i, j].pcolormesh(yhat[j, 0, :, :].T, 
                                      norm=norm,
                                      cmap="seismic")
        
        norm = colors.CenteredNorm()
        diffs = axs3[i, j].pcolormesh(error[j, 0, :, :].T, 
                                      norm=norm,
                                      cmap="seismic")

        sample_l1_loss = round(l1_loss[j], 1)
        fd = {"fontsize" : 8}
        axs1[i, j].set_title("L: {:n} V {:n} D {:n}".format(sample_l1_loss, v, d), fd)
        axs2[i, j].set_title("L: {:n} V {:n} D {:n}".format(sample_l1_loss, v, d), fd)
        axs3[i, j].set_title("L: {:n} V {:n} D {:n}".format(sample_l1_loss, v, d), fd)
        
        # Colorbars
        fig1.colorbar(labels, ax=axs1[i, j])
        fig2.colorbar(preds, ax=axs2[i, j])
        fig3.colorbar(diffs, ax=axs3[i, j])

print("Saving plots")
fig1.savefig(runloc + "TestLabels.png")
fig2.savefig(runloc + "TestPredictions.png")
fig3.savefig(runloc + "TestError.png")