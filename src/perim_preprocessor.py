"""
This script is used for gridded PERIM pre-processing.
DATA SOURCE: PERIM
"""

import fnmatch
import os
import warnings

import numpy as np
import pandas as pd
from netCDF4 import Dataset

warnings.simplefilter(action="ignore")


def preprocessor(filename, fire_id, time, lat_lim, lon_lim):
    namelist = pd.read_csv("./input/namelist", header=None, delimiter="=")
    namelist = namelist[1]
    path_frp = str(namelist[27].replace(" ", ""))

    f_output = "./input/" + fire_id + "/" + time + "/" + filename + "." + time + ".nc"
    if os.path.isfile(f_output) is True:
        return

    date = time[:8]
    hour = time[8:10]
    
    # ---- Reading Data ----
    fname = (
        "CONUS|"
        + fire_id
        + "|"
        + date[:4]
        + "-"
        + date[4:6]
        + "-"
        + date[6:]
        + "T"
        + hour
        + ":00:00.nc"
    )
    f_ori = [
        f
        for f in os.listdir(path_frp)
        if fnmatch.fnmatch(f, fname)
    ][0]
    f_ori = path_frp + "/" + f_ori

    if os.path.isfile(f_ori) is True:
        readin = Dataset(f_ori)
        yt = np.flip(readin["y"][:])
        xt = readin["x"][:]
        xt[xt < 0] = xt[xt < 0] + 360

        data = np.squeeze(readin["fire"][:, :])
        data = np.flipud(data)
        data = np.array(data)

        # ---- ENSURE OUTPUT COVERS FULL REQUESTED BOUNDS ----
        # Get grid spacing (positive values, assuming regular grid)
        dlat = np.abs(np.median(np.diff(yt)))
        dlon = np.abs(np.median(np.diff(xt)))
        
        # Pad latitude coordinates and data if source doesn't cover lower bound
        if yt[0] > lat_lim[0]:
            n_pad = int(np.ceil((yt[0] - lat_lim[0]) / dlat))
            yt_pad = yt[0] - np.arange(n_pad, 0, -1) * dlat
            yt = np.concatenate([yt_pad, yt])
            data_pad = np.full((n_pad, data.shape[1]), 0, dtype=data.dtype)
            data = np.concatenate([data_pad, data], axis=0)
        
        # Pad latitude if source doesn't cover upper bound
        if yt[-1] < lat_lim[1]:
            n_pad = int(np.ceil((lat_lim[1] - yt[-1]) / dlat))
            yt_pad = yt[-1] + np.arange(1, n_pad + 1) * dlat
            yt = np.concatenate([yt, yt_pad])
            data_pad = np.full((n_pad, data.shape[1]), 0, dtype=data.dtype)
            data = np.concatenate([data, data_pad], axis=0)
        
        # Pad longitude coordinates and data if source doesn't cover lower bound
        if xt[0] > lon_lim[0]:
            n_pad = int(np.ceil((xt[0] - lon_lim[0]) / dlon))
            xt_pad = xt[0] - np.arange(n_pad, 0, -1) * dlon
            xt = np.concatenate([xt_pad, xt])
            data_pad = np.full((data.shape[0], n_pad), 0, dtype=data.dtype)
            data = np.concatenate([data_pad, data], axis=1)
        
        # Pad longitude if source doesn't cover upper bound
        if xt[-1] < lon_lim[1]:
            n_pad = int(np.ceil((lon_lim[1] - xt[-1]) / dlon))
            xt_pad = xt[-1] + np.arange(1, n_pad + 1) * dlon
            xt = np.concatenate([xt, xt_pad])
            data_pad = np.full((data.shape[0], n_pad), 0, dtype=data.dtype)
            data = np.concatenate([data, data_pad], axis=1)
        
        # ---- Create Coordinate Grids ----
        xt_grid, yt_grid = np.meshgrid(xt, yt)

        readin.close()
        del [readin, dlat, dlon]

    else:
        return 1

    # ---- Write NetCDF File ----
    f = Dataset(f_output, "w")
    f.createDimension("time", 1)
    f.createDimension("nlat", data.shape[0])
    f.createDimension("nlon", data.shape[1])

    var_input = f.createVariable("fire", "float", ("time", "nlat", "nlon"))
    var_lat = f.createVariable("grid_lat", "float", ("nlat", "nlon"))
    var_lon = f.createVariable("grid_lon", "float", ("nlat", "nlon"))
    var_time = f.createVariable("time", str, ("time",))

    var_input[:] = data
    var_lat[:] = yt_grid
    var_lon[:] = xt_grid
    var_time[:] = np.array(time).astype(str)
    f.close()

    return 0
