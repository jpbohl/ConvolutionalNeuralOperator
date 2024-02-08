import torch
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import json
import sys
from statistics import mean, stdev

from Problems.StrakaMB import StrakaDataset as StrakaMB
from Problems.Straka import StrakaDataset
from CNOModule import CNO
import torch


# Getting arguments for the timestep at which model was trained and
# the names of the runs to plot
time = 900
run = "light-gorge-23"
cno = True

runloc = f"/cluster/work/math/jabohl/runs/{run}/"

device = "cpu"

model = torch.load(runloc + "model.pkl", map_location=device)
model.to(device)

dataloc = "/cluster/work/math/jabohl/StrakaMovingBubble/data/"
dataset = StrakaMB(dataloc, time=time, s=256, training_samples=400, which="validation", cno=cno, cluster=False)

loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
model.eval()
rel_l1_loss = 0.

with torch.no_grad():
    for x, y in loader:

        pred = model(x)

        error = y - pred
        error = error

        rel_l1_loss += (torch.mean(abs(error)) / torch.mean(abs(y)) * 100)
    
rel_l1_loss /= len(loader)

print(rel_l1_loss)
