import torch
import matplotlib.pyplot as plt
import numpy as np
import json

from Problems.Straka import Straka, StrakaFNO


# Loading config files
with open("cno_architecture.json", "r") as f:
    cno_config = json.loads(f.read())
with open("fno_architecture.json", "r") as f:
    fno_config = json.loads(f.read())
with open("training_properties.json", "r") as f:
    train_config = json.loads(f.read())


# Loading FNO Model and Data

dataloc = "/cluster/work/math/jabohl/Straka_ensemble_vd/data/"
problem = StrakaFNO(fno_config, "cuda", 4, 20, dataloc=dataloc)

saved_model = torch.load("/cluster/work/math/jabohl/FNO_Straka/model.pkl")

xs, ys = next(iter(problem.test_loader))

saved_model.eval()
predict = saved_model(xs.to("cuda"))

predict_fno = predict.cpu().detach().numpy()


# Loading CNO Model and Data
dataloc = "/cluster/work/math/jabohl/Straka_ensemble_vd/data/"
problem = Straka(cno_config, "cuda", 4, 20, dataloc=dataloc)
saved_model = torch.load("/cluster/work/math/jabohl/CNO_Straka/model.pkl")

xs, ys = next(iter(problem.test_loader))

saved_model.eval()
predict = saved_model(xs.to("cuda"))


predict_cno = predict.cpu().detach().numpy()


# Plotting predictions
fig, axes = plt.subplots(1, 3, sharey=True)

labels = axes[0].pcolormesh(ys[0, 0, :, :].T, cmap="Blues_r")
axes[0].set_title("Labels")


axes[1].pcolormesh(predict_fno[0,:,  :, 0].T, cmap="Blues_r")
axes[1].set_title("FNO")

axes[2].pcolormesh(predict_cno[0, 0,  :, :].T, cmap="Blues_r")
axes[2].set_title("CNO")


fig.colorbar(labels, ax=axes.ravel().tolist())
fig.savefig("Predictions.png")
