import os
import random
import json
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

#Poisson data:

class StrakaDataset(Dataset):
    def __init__(self, dataloc, which="training", nf=0, training_samples = 400, time=300, s=128, ntest=None, in_dist = True, cno=True, cluster=True, time0=0):
        
        # Get files:
        with open(dataloc + "files.json", "r") as f:
            files = json.loads(f.read())

        # Determine number of total samples
        total_samples = len(files)

        if not ntest:
            ntest = (total_samples - training_samples) // 2

        self.files_t0 = [dataloc + f + f"/fields/{time0}.nc" for f in files[:total_samples]]
        self.files_t1 = [dataloc + f + f"/fields/{time}.nc" for f in files[:total_samples]]
                    
        drop_t = ["u", "w", "v", "s", "buoyancy_frequency"]
        drop_wind = ["v", "temperature", "s", "buoyancy_frequency"]

        parallel = True if cluster else False
        self.t0 = xr.open_mfdataset(self.files_t0, combine="nested", concat_dim="index", parallel=True, drop_variables=drop_t, autoclose=True).temperature                    
        self.t1 = xr.open_mfdataset(self.files_t1, combine="nested", concat_dim="index", parallel=True, drop_variables=drop_t, autoclose=True).temperature
        wind = xr.open_mfdataset(self.files_t1, combine="nested", concat_dim="index", parallel=True, drop_variables=drop_wind, autoclose=True)

        # vorticity 
        new_zs = np.linspace(0, 6400, s)
        wind = wind.isel(x=np.arange(511, 1023)).coarsen(x=2).mean()
        wind = wind.interp(z=new_zs, kwargs={"fill_value":"extrapolate"})

        dwdx = wind.w.differentiate("x")
        dudz = wind.u.differentiate("z")
        self.vorticity = dwdx - dudz

        # Background profile 
        self.bpf = self.t0.isel(index=0, x=0).data
        
        # Removing background profile
        self.t0 = self.t0 - self.bpf
        self.t1 = self.t1 - self.bpf
        
        # Selecting windows of interest
        new_zs = np.linspace(0, 6400, s)

        # interpolate in z and coarsen in x
        self.t0 = self.t0.isel(x=np.arange(511, 511+s)).interp(z=new_zs, kwargs={"fill_value": "extrapolate"})
        self.t1 = self.t1.isel(x=np.arange(511, 1023)).coarsen(x=2).mean()
        self.t1 = self.t1.interp(z=new_zs, kwargs={"fill_value":"extrapolate"})


        if cluster:
            # Write files to tmp
            try:
                TMP = os.environ["TMPDIR"]
            except KeyError:
                TMP = ""

            # Save data to avoid interpolating every time a sample is loaded
            self.t0.to_netcdf(TMP + "t0.nc", mode="w")
            self.t1.to_netcdf(TMP + "t1.nc", mode="w")
            self.vorticity.to_netcdf(TMP + "vorticity.nc", mode="w")

            # Reload data
            self.t0 = xr.open_dataarray(TMP + "t0.nc", engine="netcdf4")
            self.t1 = xr.open_dataarray(TMP + "t1.nc", engine="netcdf4")
            self.vorticity = xr.open_dataarray(TMP + "vorticity.nc", engine="netcdf4")

        # Computing stats for standardization
        self.mean_ic = self.t0.mean().compute()
        self.std_ic = self.t0.std().compute()

        self.mean_data = torch.tensor(self.mean_ic.item())
        self.std_data = torch.tensor(self.std_ic.item())
        
        self.mean_model = self.t1.mean().compute().item()
        self.std_model = self.t1.std().compute().item()

        self.s = s #Sampling rate
        self.ntest = ntest

        self.cno = cno # Determines if dataset is used for cno or fno

        # Splitting data into train, test and validation sets
        if which == "training":
            self.length = training_samples
            self.start = 0

            if training_samples > total_samples:
                raise ValueError("Only {self.nsamples} samples available")

        elif which == "validation":
            self.length = self.ntest
            self.start = training_samples 

        elif which == "test":
            if in_dist: #Is it in-distribution?
                self.length = self.ntest
                self.start = training_samples + self.ntest
            else:
                raise NotImplementedError("Out of distribution training not implemented")
        
        #Fourier modes (Default is 0):
        self.N_Fourier_F = nf

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        index = index + self.start
        
        # Assembling inputs
        inputs = torch.tensor(self.t0.isel(index=index).compute().data, dtype=torch.float32)

        # Getting labels and vorticity
        labels = torch.tensor(self.t1.isel(index=index).compute().data, dtype=torch.float32)
        vorticity = torch.tensor(self.vorticity.isel(index=index).compute().data, dtype=torch.float32)
        weights = 1000 * abs(vorticity) + 1

        # Standardize data
        inputs = (inputs - self.mean_data) / self.std_data
        labels = (labels - self.mean_model) / self.std_model

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

        return inputs, labels, weights

    def get_grid(self):
        x = torch.linspace(0, 1, self.s)
        y = torch.linspace(0, 1, self.s)
        x_grid, y_grid = torch.meshgrid(x, y)
        x_grid = x_grid.unsqueeze(-1)
        y_grid = y_grid.unsqueeze(-1)
        grid = torch.cat((x_grid, y_grid), -1)
        return grid

class Straka:
    def __init__(self, network_properties, device, batch_size, training_samples = 400,time=300, time0=0, s = 128, ntest=None, in_dist = True, dataloc="data/", cluster=True):
        
        #Must have parameters: ------------------------------------------------        

        if "in_size" in network_properties:
            self.in_size = network_properties["in_size"]
        else:
            raise ValueError("You must specify the computational grid size.")
        
        if "out_size" in network_properties:
            self.out_size = network_properties["out_size"]
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
        attention = network_properties["attention"]
    
        torch.manual_seed(retrain)
        
        #----------------------------------------------------------------------

        self.model = CNO(in_dim=1 + 2 * self.N_Fourier_F,  # Number of input channels.
                                in_size=self.in_size,
                                out_size=self.out_size,
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
                                attention=attention
                                ).to(device)
        
        #Change number of workers accoirding to your preference
        if cluster:
            num_workers = 8
        else:
            num_workers = 0

        self.train_loader = DataLoader(StrakaDataset(dataloc, "training", self.N_Fourier_F, training_samples, time, s,  time0=time0, cluster=cluster), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(StrakaDataset(dataloc, "validation", self.N_Fourier_F, training_samples, time, s,  time0=time0, cluster=cluster), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(StrakaDataset(dataloc, "test", self.N_Fourier_F, training_samples, time, s, in_dist=in_dist,  time0=time0, cluster=cluster), batch_size=batch_size, shuffle=False, num_workers=num_workers)


class StrakaFNO:
    def __init__(self, network_properties, device, batch_size, training_samples = 400, time=300, s = 128, in_dist = True, dataloc="data/", cluster=True):
        
        retrain = network_properties["retrain"]

        if "in_size" in network_properties:
            s = network_properties["in_size"]

        torch.manual_seed(retrain)

        if "FourierF" in network_properties:
            self.N_Fourier_F = network_properties["FourierF"]
        else:
            self.N_Fourier_F = 0
        
        self.model = FNO2d(network_properties, device, 0, 1+ 2 * self.N_Fourier_F)

        #----------------------------------------------------------------------  

        if cluster:
            num_workers = 8
        else:
            num_workers = 0
        
        self.train_loader = DataLoader(StrakaDataset(dataloc, "training", self.N_Fourier_F, training_samples, time=time, s=s, cno=False, cluster=cluster), 
                                batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(StrakaDataset(dataloc, "validation", self.N_Fourier_F, training_samples, time=time, s=s, cno=False, cluster=cluster), 
                                batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(StrakaDataset(dataloc, "test", self.N_Fourier_F, training_samples, time=time, s=s, in_dist=in_dist, cno=False, cluster=cluster), 
                                batch_size=batch_size, shuffle=False, num_workers=num_workers)