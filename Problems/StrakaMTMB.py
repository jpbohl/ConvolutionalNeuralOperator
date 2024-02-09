import random
import json
import os
from statistics import mean, stdev

import xarray as xr
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from CNOModule import CNO
from FNOModules import FNO2d
from torch.utils.data import Dataset

import xarray

from training.FourierFeatures import FourierFeatures

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

#------------------------------------------------------------------------------

# Some functions needed for loading the Navier-Stokes data
def samples_fft(u):
    return scipy.fft.fft2(u, norm='forward', workers=-1)

def samples_ifft(u_hat):
    return scipy.fft.ifft2(u_hat, norm='forward', workers=-1).real

def downsample(u, N):
    N_old = u.shape[-2]
    freqs = scipy.fft.fftfreq(N_old, d=1/N_old)
    sel = np.logical_and(freqs >= -N/2, freqs <= N/2-1)
    u_hat = samples_fft(u)
    u_hat_down = u_hat[:,:,sel,:][:,:,:,sel]
    u_down = samples_ifft(u_hat_down)
    return u_down

#------------------------------------------------------------------------------

#Load default parameters:
    
def default_param(network_properties):
    
    if "channel_multiplier" not in network_properties:
        network_properties["channel_multiplier"] = 32
    
    if "half_width_mult" not in network_properties:
        network_properties["half_width_mult"] = 1
    
    if "lrelu_upsampling" not in network_properties:
        network_properties["lrelu_upsampling"] = 2
    
    if "res_len" not in network_properties:
        network_properties["res_len"] = 1

    if "filter_size" not in network_properties:
        network_properties["filter_size"] = 6
    
    if "radial" not in network_properties:
        network_properties["radial_filter"] = 0
    
    if "cutoff_den" not in network_properties:
        network_properties["cutoff_den"] = 2.0001
    
    if "FourierF" not in network_properties:
        network_properties["FourierF"] = 0
    
    if "retrain" not in network_properties:
         network_properties["retrain"] = 4
    
    if "kernel_size" not in network_properties:
        network_properties["kernel_size"] = 3
    
    return network_properties

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class StrakaDataset(Dataset):
    def __init__(self, dataloc, which="training", nf=0, training_samples = 400, s=256, ntest=None, in_dist = True, cno=True, cluster=True):
        
        #Overview file:
        with open(dataloc + "files.json") as f:
            self.files = json.load(f)

        total_samples = len(self.files)

        if not ntest:
            ntest = (total_samples - training_samples) // 2

        self.files_t0 = [dataloc + f + "/fields/0.nc" for f in self.files[:total_samples]]
        self.files_t1 = [dataloc + f + f"/fields/300.nc" for f in self.files[:total_samples]]
        self.files_t2 = [dataloc + f + f"/fields/600.nc" for f in self.files[:total_samples]]
        self.files_t3 = [dataloc + f + f"/fields/900.nc" for f in self.files[:total_samples]]
                    
        drop = ["u", "v", "w", "s", "buoyancy_frequency"]

        parallel = True if cluster else False
        self.t0 = xr.open_mfdataset(self.files_t0, combine="nested", concat_dim="index", parallel=True, drop_variables=drop, autoclose=True).temperature                    
        self.t1 = xr.open_mfdataset(self.files_t1, combine="nested", concat_dim="index", parallel=True, drop_variables=drop, autoclose=True).temperature
        self.t2 = xr.open_mfdataset(self.files_t2, combine="nested", concat_dim="index", parallel=True, drop_variables=drop, autoclose=True).temperature
        self.t3 = xr.open_mfdataset(self.files_t3, combine="nested", concat_dim="index", parallel=True, drop_variables=drop, autoclose=True).temperature

        # Background profile 
        self.bpf = self.t0.isel(index=0, x=0).data
        
        # Removing background profile
        self.t0 = self.t0 - self.bpf
        self.t1 = self.t1 - self.bpf
        self.t2 = self.t2 - self.bpf
        self.t3 = self.t3 - self.bpf
        
        # Interpolate in z
        new_zs = np.linspace(0, 6400, s)
        self.t0 = self.t0.isel(x=np.arange(511, 511+256)).interp(z=new_zs, kwargs={"fill_value": "extrapolate"})
        self.t1 = self.t1.isel(x=np.arange(511, 1023)).interp(z=new_zs, kwargs={"fill_value": "extrapolate"})
        self.t2 = self.t2.isel(x=np.arange(511, 1023)).interp(z=new_zs, kwargs={"fill_value": "extrapolate"})
        self.t3 = self.t3.isel(x=np.arange(511, 1023)).interp(z=new_zs, kwargs={"fill_value": "extrapolate"})
        
        # Coarsen in x direction
        self.t1 = self.t1.coarsen(x=2).mean()
        self.t2 = self.t2.coarsen(x=2).mean()
        self.t3 = self.t3.coarsen(x=2).mean()
        
        if cluster:
            # Write files to tmp
            try:
                TMP = os.environ["TMPDIR"]
            except KeyError:
                TMP = ""

            # Save data to avoid interpolating every time a sample is loaded
            self.t0.to_netcdf(TMP + "t0.nc", mode="w")
            self.t1.to_netcdf(TMP + "t1.nc", mode="w")
            self.t2.to_netcdf(TMP + "t2.nc", mode="w")
            self.t3.to_netcdf(TMP + "t3.nc", mode="w")

            # Reload data
            self.t0 = xr.open_dataarray(TMP + "t0.nc", engine="netcdf4")
            self.t1 = xr.open_dataarray(TMP + "t1.nc", engine="netcdf4")
            self.t2 = xr.open_dataarray(TMP + "t2.nc", engine="netcdf4")
            self.t3 = xr.open_dataarray(TMP + "t3.nc", engine="netcdf4")

        # Computing stats for standardization
        self.mean0 = self.t0.mean().compute().item()
        self.mean1 = self.t1.mean().compute().item()
        self.mean2 = self.t2.mean().compute().item()
        self.mean3 = self.t3.mean().compute().item()
        
        self.std0 = self.t0.std().compute().item()
        self.std1 = self.t1.std().compute().item()
        self.std2 = self.t2.std().compute().item()
        self.std3 = self.t3.std().compute().item()

        self.s = s #Sampling rate
        self.nsamples = total_samples
        self.ntest = ntest

        self.cno = cno # Determines if dataset is used for cno or fno

        # Splitting data into train, test and validation sets
        if which == "training":
            self.length = 3 * training_samples
            self.start = 0

            if training_samples > self.nsamples:
                raise ValueError("Only {self.nsamples} samples available")

        elif which == "validation":
            self.length = 3 * self.ntest
            self.start = training_samples 

        elif which == "test":
            self.length = 3 * self.ntest
            self.start = training_samples + self.ntest
        
        #Fourier modes (Default is 0):
        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        samples_per_ts = int(self.length / 3)
        if index in range(0, samples_per_ts):

            time = 0.3

            label_data = self.t1
            label_mean = self.mean1
            label_std = self.std1

            index = index + self.start

        elif index in range(samples_per_ts, 2 * samples_per_ts):

            time = 0.6

            label_data = self.t2
            label_mean = self.mean2
            label_std = self.std2

            index = index - samples_per_ts + self.start

        elif index in range(2 * samples_per_ts, self.length):

            time = 0.9

            label_data = self.t3
            label_mean = self.mean3
            label_std = self.std3

            index = index - 2 * samples_per_ts + self.start
        
        # Get initial condition
        ic = torch.tensor(self.t0.isel(index=index).compute().data, dtype=torch.float32)

        # Getting labels
        labels = torch.tensor(label_data.isel(index=index).compute().data, dtype=torch.float32)

        # Standardize data
        ic = (ic - self.mean0) / self.std0
        labels = (labels - label_mean) / label_std

        # Add time to input
        time = time * torch.ones_like(ic)
        inputs = torch.cat([time, ic], dim=-1)

        # Reshape tensors to nchannels x s x s (shape expected by CNO code)
        if self.cno:
            inputs = torch.movedim(inputs, -1, 0)
            labels = torch.movedim(labels, -1, 0)
        
        if self.N_Fourier_F > 0:
            grid = self.get_grid()
            FF = FourierFeatures(1, self.N_Fourier_F, grid.device)
            ff_grid = FF(grid)
            ff_grid = ff_grid.permute(2, 0, 1)
            inputs = torch.cat((inputs, ff_grid), 0)

        assert not inputs.isnan().any()
        assert not labels.isnan().any()

        return inputs, labels

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)
        x_grid, y_grid = torch.meshgrid(x, y)
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid

class Straka:
    def __init__(self, network_properties, device, batch_size, training_samples = 400,time=300, s = 128, ntest=56, in_dist = True, dataloc="data/", cluster=True):
        
        #Must have parameters: ------------------------------------------------        

        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
        else:
            raise ValueError("You must specify the computational grid size.")
        
        if "N_layers" in network_properties:
            N_layers = network_properties["N_layers"]
        else:
            raise ValueError("You must specify the number of (D) + (U) blocks.")
        
        if "N_res" in network_properties:
                N_res = network_properties["N_res"]        
        else:
            raise ValueError("You must specify the number of (R) blocks.")
        
        if "N_res_neck" in network_properties:
                N_res_neck = network_properties["N_res_neck"]        
        else:
            raise ValueError("You must specify the number of (R)-neck blocks.")
        
        
        #Load default parameters if they are not in network_properties
        network_properties = default_param(network_properties)
        
        
        kernel_size = network_properties["kernel_size"]
        channel_multiplier = network_properties["channel_multiplier"]
        res_len = network_properties["res_len"]
        retrain = network_properties["retrain"]
        self.N_Fourier_F = network_properties["FourierF"]
        
        #Filter properties: ---------------------------------------------------
        cutoff_den = network_properties["cutoff_den"]
        filter_size = network_properties["filter_size"]
        radial = network_properties["radial_filter"]
        half_width_mult = network_properties["half_width_mult"]
        lrelu_upsampling = network_properties["lrelu_upsampling"]
    
        torch.manual_seed(retrain)
        
        #----------------------------------------------------------------------

        self.model = CNO(in_dim=2 + 2 * self.N_Fourier_F,  # Number of input channels.
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
        
        #Change number of workers accoirding to your preference
        if cluster:
            num_workers = 8
        else:
            num_workers = 0

        self.train_loader = DataLoader(StrakaDataset(dataloc, "training", self.N_Fourier_F, training_samples, s, ntest=ntest, cluster=cluster), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(StrakaDataset(dataloc, "validation", self.N_Fourier_F, training_samples, s, ntest=ntest, cluster=cluster), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(StrakaDataset(dataloc, "test", self.N_Fourier_F, training_samples, s, in_dist=in_dist, ntest=ntest, cluster=cluster), batch_size=batch_size, shuffle=False, num_workers=num_workers)


class StrakaFNO:
    def __init__(self, network_properties, device, batch_size, training_samples = 400, s = 256, in_dist = True, dataloc="data/", cluster=True):
        
        retrain = network_properties["retrain"]
        torch.manual_seed(retrain)

        if "FourierF" in network_properties:
            self.N_Fourier_F = network_properties["FourierF"]
        else:
            self.N_Fourier_F = 0
        
        self.model = FNO2d(network_properties, device, 0, 2 + 2 * self.N_Fourier_F)

        #----------------------------------------------------------------------  

        #Change number of workers accoirding to your preference
        if cluster:
            num_workers = 8
        else:
            num_workers = 0
        
        self.train_loader = DataLoader(StrakaDataset(dataloc, "training", self.N_Fourier_F, training_samples, s=s, cno=False, cluster=cluster), 
                                batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(StrakaDataset(dataloc, "validation", self.N_Fourier_F, training_samples, s=s, cno=False, cluster=cluster), 
                                batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(StrakaDataset(dataloc, "test", self.N_Fourier_F, training_samples, s=s, in_dist=in_dist, cno=False, cluster=cluster), 
                                batch_size=batch_size, shuffle=False, num_workers=num_workers)
