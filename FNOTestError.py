import torch
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import json
import sys
from statistics import mean, stdev

from Problems.Straka import StrakaDataset, Straka, StrakaFNO
from Problems.StrakaMB import StrakaDataset as StrakaMB

from FNOModules import FNO2d

# Getting arguments for the timestep at which model was trained and
# the names of the runs to plot
time = 900
run = "dashing-water-9"
cno = False

runloc = f"/cluster/work/math/jabohl/runs/{run}/"

# Loading config files
with open(runloc + "net_architecture.json", "r") as f:
    fno_config = json.loads(f.read())

with open(runloc + "training_properties.json", "r") as f:
    train_config = json.loads(f.read())

# Loading Model and Data
device = "cpu"

dataloc = "/cluster/work/math/jabohl/StrakaMovingBubble/data/"
dataset = StrakaMB(dataloc, time=time, training_samples=400, s=256, which="validation", cno=cno, cluster=True)

model = FNO2d(fno_config, device, 0, 1)
weights = torch.load(runloc + "model_weights.pt", map_location=device)
model.load_state_dict(weights)

loader = torch.utils.data.DataLoader(dataset, batch_size=16)

model.eval()
model.to("cpu")
rel_l1_loss = 0.

with torch.no_grad():
    for x, y in loader:

        x.to(device)
        y.to(device)
        pred = model(x)

        error = y - pred
        rel_l1_loss += (torch.mean(abs(error), axis=(0,1,2,3)) / torch.mean(abs(y), axis=(0,1,2,3)) * 100)
    
rel_l1_loss /= len(loader)

print(rel_l1_loss)