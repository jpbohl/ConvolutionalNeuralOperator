import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

from Problems.Straka import StrakaDataset, Straka, StrakaFNO

# Getting arguments for the timestep at which model was trained and
# the names of the runs to plot
time = int(sys.argv[1])
fno_run = sys.argv[2]
cno_run = sys.argv[3]

# Loading config files
with open("cno_architecture.json", "r") as f:
    cno_config = json.loads(f.read())
with open("fno_architecture.json", "r") as f:
    fno_config = json.loads(f.read())
with open("training_properties.json", "r") as f:
    train_config = json.loads(f.read())

# Loading FNO Model and Data
device = "cuda" if torch.cuda.is_available() else "cpu"

dataloc = "/cluster/work/math/jabohl/StrakaData/data/"
dataset = StrakaDataset(dataloc, training_samples=10, time=time, ntest=56, which="test", cno=False)
loader = torch.utils.data.DataLoader(dataset, batch_size=8)

saved_model = torch.load(f"/cluster/work/math/jabohl/runs/{fno_run}/model.pkl")

# Get batch of test samples
xs, ys = next(iter(loader))

# Computing prediction and difference between labels and prediction
saved_model.eval()
predict = saved_model(xs.to(device))

predict_fno = predict.cpu().detach().numpy()
error_fno = ys.numpy() - predict_fno

# Loading CNO Model and Data
dataloc = "/cluster/work/math/jabohl/StrakaData/data/"
dataset = StrakaDataset(dataloc, training_samples=10, time=time, ntest=56, which="test")
loader = torch.utils.data.DataLoader(dataset, batch_size=8)

saved_model = torch.load(f"/cluster/work/math/jabohl/runs/{cno_run}/model.pkl")

# Get batch of test samples
xs, ys = next(iter(loader))

# Computing prediction and difference between labels and prediction
saved_model.eval()
predict = saved_model(xs.to(device))

predict_cno = predict.cpu().detach().numpy()
error_cno = ys.numpy() - predict_cno


# Plotting predictions
fig, axes = plt.subplots(1, 3, sharey=True)

labels = axes[0].pcolormesh(ys[0, 0, :, :].T, cmap="Blues_r")
axes[0].set_title("Labels")
fig.colorbar(labels, ax=axes[0])

fno = axes[1].pcolormesh(predict_fno[0,:,  :, 0].T, cmap="Blues_r")
axes[1].set_title("FNO")
fig.colorbar(fno, ax=axes[1])

cno = axes[2].pcolormesh(predict_cno[0, 0,  :, :].T, cmap="Blues_r")
axes[2].set_title("CNO")
fig.colorbar(cno, ax=axes[2])

fig.savefig("Predictions.png")

# Plotting errors
fig, axes = plt.subplots(1, 3, sharey=True)

labels = axes[0].pcolormesh(ys[0, 0, :, :].T, cmap="Blues_r")
axes[0].set_title("Labels")
fig.colorbar(labels, ax=axes[0])

error_fno = axes[1].pcolormesh(error_fno[0,:,  :, 0].T, cmap="Blues_r")
axes[1].set_title("FNO Error")
fig.colorbar(error_fno, ax=axes[1])

error_cno = axes[2].pcolormesh(error_cno[0, 0,  :, :].T, cmap="Blues_r")
axes[2].set_title("CNO Error")
fig.colorbar(error_cno, ax=axes[2])

fig.savefig("Error.png")
