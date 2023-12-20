import torch
import matplotlib.pyplot as plt
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
    fno_config = json.loads(f.read())

with open("cno_training_properties.json", "r") as f:
    train_config = json.loads(f.read())

# Loading Model and Data
device = "cpu"

dataloc = "/cluster/work/math/jabohl/StrakaData/data/"
dataset = StrakaDataset(dataloc, time=time, ntest=56, which="validation", cno=cno)

model = torch.load(runloc + "model.pkl", map_location=device)
model = model.to(device)
model.device = device

# Adding attention attribute for models which did not have attention
if not hasattr(model, "self_attention"):
    print("Adding attention attribute to model")
    model.attention = False
    model.self_attention = torch.nn.Identity()
else:
    model.attention = True

# Computing prediction and difference between labels and prediction
model.eval()

# Plotting Testset Labels, Predictions and Error
nrows = 4
ncols = 4
loader = iter(torch.utils.data.DataLoader(dataset, batch_size=4))
fig1, axs1 = plt.subplots(nrows, ncols, sharex = True, sharey=True)
fig2, axs2 = plt.subplots(nrows, ncols, sharex = True, sharey=True)
fig3, axs3 = plt.subplots(nrows, ncols, sharex = True, sharey=True)

fig1.subplots_adjust(hspace=.5)
fig2.subplots_adjust(hspace=.5)
fig3.subplots_adjust(hspace=.5)

for i in range(nrows):
    x, y = next(loader) # Get batch from test loader
    y = y.detach().cpu().numpy()

    # Make prediction and compute error
    yhat = model(x.to(device)).detach().cpu().numpy()
    error = y - yhat

    # Relative L1 Error
    l1_loss = np.mean(abs(yhat - y), axis=(1,2,3)) / np.mean(abs(y)) * 100

    for j in range(ncols):
        
        # Getting viscosity and density
        v = round(x[j, 0, 0, 0].item(), 1)
        d = round(x[j, 1, 0, 0].item(), 1)

        # Plotting
        labels = axs1[i, j].pcolormesh(y[j, 0, :, :].T)
        preds = axs2[i, j].pcolormesh(yhat[j, 0, :, :].T)
        diffs = axs3[i, j].pcolormesh(error[j, 0, :, :].T)

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
