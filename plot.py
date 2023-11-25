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
device = "cuda" if torch.cuda.is_available() else "cpu"
#dataloc = "/cluster/work/math/jabohl/Straka_ensemble_vd/data/"
dataloc = "/Users/jan/sempaper/straka_data/"
problem = StrakaFNO(fno_config, device, 2, 2,time=600, s=256,dataloc=dataloc)

#saved_model = torch.load("/cluster/work/math/jabohl/FNO_Straka/model.pkl")
saved_model = torch.load("/Users/jan/sempaper/ConvolutionalNeuralOperator/TrainedModels/FNO_straka/model.pkl")

xs, ys = next(iter(problem.test_loader))

saved_model.eval()
predict = saved_model(xs.to(device))

predict_fno = predict.cpu().detach().numpy()

# Loading CNO Model and Data
#dataloc = "/cluster/work/math/jabohl/Straka_ensemble_vd/data/"
dataloc = "/Users/jan/sempaper/straka_data/"
problem = Straka(cno_config, device, 2, 2, time=600,s=256,dataloc=dataloc)
#saved_model = torch.load("/cluster/work/math/jabohl/CNO_Straka/model.pkl")
saved_model = torch.load("/Users/jan/sempaper/ConvolutionalNeuralOperator/TrainedModels/CNO_straka/model.pkl")

xs, ys = next(iter(problem.test_loader))

saved_model.eval()
predict = saved_model(xs.to(device))


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