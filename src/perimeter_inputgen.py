"""
This script is used to generate model input files. Pre-process gridded fire map needed.
"""

import os
import warnings
import logging
from datetime import datetime
import glob
from math import floor, ceil

import numpy as np
import pandas as pd
import xarray as xr
from metpy.calc import wind_direction
from metpy.units import units
from netCDF4 import Dataset
from scipy import ndimage
from scipy.interpolate import griddata

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

warnings.simplefilter(action="ignore")


def file_finder(item, date, initial_hour, hour):
    logger.debug(f"Finding file for item: {item}, date: {date}, initial_hour: {initial_hour}, hour: {hour}")
    namelist = pd.read_csv("./input/namelist", header=None, delimiter="=")
    namelist = namelist[1]

    if item == "frp":
        return (
            str(namelist[21].replace(" ", ""))
            + "/RAVE-HrlyEmiss-3km_v2r0_blend_s"
            + date
            + ("%02d" % initial_hour)
            + "00000*.nc"
        )
    elif item == "elv":
        return str(namelist[22].replace(" ", "")) + "/MERIT_DEM.nc"
    elif item == "ast":
        return str(namelist[23].replace(" ", "")) + "/VIIRS_AST_2024.nc"
    elif item == "fh":
        return str(namelist[24].replace(" ", "")) + "/GLAD.nc"
    elif item == "vhi":
        week = datetime(int(date[:4]), int(date[4:6]), int(date[6:])).isocalendar().week
        lastweek = datetime(int(date[:4]), 12, 31).isocalendar().week
        if week == lastweek:
            return (
                str(namelist[25].replace(" ", ""))
                + "/npp/VHP.G04.C07.npp.P2025"
                + ("%03d" % (lastweek - 1))
                + ".VH.nc",
                str(namelist[25].replace(" ", ""))
                + "/j01/VHP.G04.C07.j01.P2025"
                + ("%03d" % (lastweek - 1))
                + ".VH.nc",
            )
        else:
            return (
                str(namelist[25].replace(" ", ""))
                + "/npp/VHP.G04.C07.npp.P2020"
                + ("%03d" % week)
                + ".VH.nc",
                str(namelist[25].replace(" ", ""))
                + "/j01/VHP.G04.C07.j01.P2020"
                + ("%03d" % week)
                + ".VH.nc",
            )
    elif (item == "t2m") or (item == "sh2") or (item == "wd") or (item == "ws"):
        return (
            str(namelist[26].replace(" ", ""))
            + "/"
            + date[:8]
            + "/hrrr.t"
            + ("%02d" % initial_hour)
            + "z.wrfsfcf"
            + ("%02d" % hour)
            + ".grib2"
        )
    elif item == "prate":
        return (
            str(namelist[26].replace(" ", ""))
            + "/"
            + date[:8]
            + "/hrrr.t"
            + ("%02d" % initial_hour)
            + "z.wrfsfcf"
            + ("%02d" % hour)
            + ".grib2"
        )


def mapping(xgrid, ygrid, data, xdata, ydata, map_method, fill_value):
    logger.debug(f"Mapping data using method: {map_method}, fill_value: {fill_value}")
    logger.debug(f"Input data shape: {data.shape}, grid shapes: {xgrid.shape}, {ygrid.shape}")
    output = griddata(
        (xdata, ydata), data, (xgrid, ygrid), method=map_method, fill_value=fill_value
    )
    logger.debug(f"Output mapping shape: {output.shape}, NaN count: {np.isnan(output).sum()}")
    return output


def wind_conversion(
    LAT, U, V
):  # from HRRR FAQ (https://rapidrefresh.noaa.gov/faq/HRRR.faq.html)
    logger.debug("Converting wind components for Lambert conformal projection")
    rotcon_p = 0.622515
    lat_tan_p = 38.5

    angle = rotcon_p * (LAT - lat_tan_p) * 0.017453  # LAMBERT CONFORMAL PROJECTION
    sinx2 = np.sin(angle)
    cosx2 = np.cos(angle)

    U_new = cosx2 * U + sinx2 * V
    V_new = (-1) * sinx2 * U + cosx2 * V
    logger.debug(f"Wind conversion completed. U range: [{U_new.min():.3f}, {U_new.max():.3f}], "
                 f"V range: [{V_new.min():.3f}, {V_new.max():.3f}]")
    return U_new, V_new


def normalization(var, NN):
    logger.debug(f"Calculating normalization coefficients for {NN} variables")
    a = []
    b = []
    for i in np.arange(NN):
        var_min = np.nanmin(var[:, :, :, i])
        var_max = np.nanmax(var[:, :, :, i])
        a = np.append(a, var_max - var_min)
        b = np.append(b, var_min)
        logger.debug(f"Variable {i}: min={var_min:.4f}, max={var_max:.4f}, range={var_max-var_min:.4f}")
    return a, b


def main_driver(initial_hour, forecast_hour, f_input, f_output, lat_lim, lon_lim):
    logger.info(f"Starting main_driver: initial_hour={initial_hour}, forecast_hour={forecast_hour}, "
                f"input={f_input}, output={f_output}, lat_lim={lat_lim}, lon_lim={lon_lim}")

    if os.path.exists(f_output):
        logger.warning(f"Output file {f_output} already exists. Skipping...")
        return 0
    
    namelist = pd.read_csv("./input/namelist", header=None, delimiter="=")
    namelist = namelist[1]
    manual_fire = int(namelist[17])
    logger.debug(f"Manual fire setting: {manual_fire}")

    # ---- Global Settings ----
    # constants
    fsize = 67  # extend fire grid

    # variable list
    firelist = ["fire", "frp"]
    geolist = ["elv", "ast", "doy", "hour"]
    veglist = ["fh", "vhi"]
    metlist = ["t2m", "sh2", "prate", "wd", "ws"]
    INPUTLIST = firelist + geolist + veglist + metlist
    logger.info(f"Input variables: {INPUTLIST}")

    # ---- Reading Fire Map & Input Settings ----
    logger.info(f"Reading fire map from: {f_input}")
    if forecast_hour == 0:  # initial fire
        readin = Dataset(f_input)
        time = readin["time"][0]
        LAT = readin["grid_lat"][:]
        LON = readin["grid_lon"][:]
        FIRE = readin["fire"][0, :, :]
        readin.close()
        del readin
        logger.info(f"Initial fire map loaded: time={time}, LAT shape={LAT.shape}, "
                   f"FIRE non-zero pixels: {(FIRE != 0).sum()}")
    else:
        readin = Dataset(f_input)
        time = readin["time"][0]
        LAT = readin["grid_lat"][:]
        LON = readin["grid_lon"][:]
        FIRE = readin["fire_pred"][0, :, :]
        readin.close()
        del readin
        logger.info(f"Forecast fire map loaded: time={time}, LAT shape={LAT.shape}, "
                   f"FIRE non-zero pixels: {(FIRE != 0).sum()}")

    NN = len(INPUTLIST)
    tt = time  # yyyymmddHHMM
    dd = time[:8]  # yyyymmdd
    hh = time[8:10]  # HH

    INPUT = np.empty([LAT.shape[0], LAT.shape[1], NN])
    logger.info(f"Selecting data for date: {dd}, hour: {hh}Z, input array shape: {INPUT.shape}")

    # ---- Reading Input Variables ----
    # perimeter
    INPUT[:, :, INPUTLIST.index("fire")] = np.copy(FIRE)
    logger.debug("FIRE data copied to input array")

    # rave
    logger.info("Processing rave data (frp)")
    filename = file_finder("frp", dd, initial_hour, forecast_hour)

    matches = glob.glob(filename)
    if matches:
        filename = matches[0]
    else:
        logger.error(f"Incorrect input file: frp - File not found: {filename}")
        return 1

    logger.debug(f"Reading frp from: {filename}")
    readin = Dataset(filename)
    yt = np.flip(readin["grid_latt"][:, 0])
    xt = readin["grid_lont"][0, :]

    data = np.squeeze(readin["FRP_MEAN"][0, :, :])
    data = np.flipud(data)
    data = np.array(data)  # fill value = -1
    data[data == -1] = 0

    qa = readin["QA"][0, :, :]
    qa = np.flipud(qa)
    data[qa == 1] = 0  # use QA = 2 and 3 only

    index1 = np.squeeze(np.argwhere((yt >= lat_lim[0]) & (yt <= lat_lim[1])))
    index2 = np.squeeze(np.argwhere((xt >= lon_lim[0]) & (xt <= lon_lim[1])))

    logger.debug(f"Rave data original shape: {readin['FRP_MEAN'][:].shape}")
    logger.debug(f"Latitude indices: {index1.shape}, Longitude indices: {index2.shape}")

    if (index1[0] == 0) & (index2[0] == 0):
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        data = data[index1[0] : index1[-1] + 2, index2[0] : index2[-1] + 2]
    elif index1[0] == 0:
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        data = data[index1[0] : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]
    elif index2[0] == 0:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        data = data[index1[0] - 1 : index1[-1] + 2, index2[0] : index2[-1] + 2]
    else:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        data = data[index1[0] - 1 : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]
    data[data < 0] = 0
    logger.debug(f"Rave data after subsetting: shape={data.shape}, "
                f"range=[{data.min():.2f}, {data.max():.2f}]")

    xt_grid, yt_grid = np.meshgrid(xt, yt)
    data_grid = mapping(
        LAT, LON, data.flatten(), yt_grid.flatten(), xt_grid.flatten(), "linear", np.nan
    )
    data_grid[data_grid < 0] = np.nan

    INPUT[:, :, INPUTLIST.index("frp")] = np.copy(data_grid)
    logger.info(f"Rave processing completed. NaN count: {np.isnan(data_grid).sum()}")

    readin.close()
    del [filename, readin, yt, xt, yt_grid, xt_grid, index1, index2, data, data_grid]



    # elv
    logger.info("Processing elevation data (elv)")
    filename = file_finder("elv", dd, initial_hour, forecast_hour)

    if os.path.isfile(filename) is False:
        logger.error(f"Incorrect input file: elv - File not found: {filename}")
        return 1

    logger.debug(f"Reading elevation from: {filename}")
    readin = Dataset(filename)
    yt = readin["lat"][:]
    xt = readin["lon"][:]
    xt[xt < 0] = xt[xt < 0] + 360
    index1 = np.squeeze(np.argwhere((yt >= lat_lim[0]) & (yt <= lat_lim[1])))
    index2 = np.squeeze(np.argwhere((xt >= lon_lim[0]) & (xt <= lon_lim[1])))

    logger.debug(f"Elevation data original shape: {readin['Band1'].shape}")
    logger.debug(f"Latitude indices: {index1.shape}, Longitude indices: {index2.shape}")

    if (index1[0] == 0) & (index2[0] == 0):
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        data = readin["Band1"][index1[0] : index1[-1] + 2, index2[0] : index2[-1] + 2]
    elif index1[0] == 0:
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        data = readin["Band1"][
            index1[0] : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2
        ]
    elif index2[0] == 0:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        data = readin["Band1"][
            index1[0] - 1 : index1[-1] + 2, index2[0] : index2[-1] + 2
        ]
    else:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        data = readin["Band1"][
            index1[0] - 1 : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2
        ]
    data[data < 0] = 0
    logger.debug(f"Elevation data after subsetting: shape={data.shape}, "
                f"range=[{data.min():.2f}, {data.max():.2f}]")

    xt_grid, yt_grid = np.meshgrid(xt, yt)
    data_grid = mapping(
        LAT, LON, data.flatten(), yt_grid.flatten(), xt_grid.flatten(), "linear", 0
    )
    data_grid[data_grid < 0] = 0

    INPUT[:, :, INPUTLIST.index("elv")] = np.copy(data_grid)
    logger.info(f"Elevation processing completed. NaN count: {np.isnan(data_grid).sum()}")

    readin.close()
    del [filename, readin, yt, xt, yt_grid, xt_grid, index1, index2, data, data_grid]

    # ast
    logger.info("Processing surface type data (ast)")
    filename = file_finder("ast", dd, initial_hour, forecast_hour)

    if os.path.isfile(filename) is False:
        logger.error(f"Incorrect input file: ast - File not found: {filename}")
        return 1

    logger.debug(f"Reading surface type from: {filename}")
    readin = Dataset(filename)
    yt = np.flip(readin["lat"][:])
    xt = readin["lon"][:]
    yt = np.round(yt, 3)
    xt = np.round(xt, 3)
    xt[xt < 0] = xt[xt < 0] + 360
    index1 = np.squeeze(np.argwhere((yt >= lat_lim[0]) & (yt <= lat_lim[1])))
    index2 = np.squeeze(np.argwhere((xt >= lon_lim[0]) & (xt <= lon_lim[1])))
    len1 = len(index1)

    # ast, hour x lat x lon
    if (index1[0] == 0) & (index2[0] == 0):
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        data = readin["surface_type"][
            len1 - index1[-1] - 2 : len1 - index1[0], index2[0] : index2[-1] + 2
        ]
    elif index1[0] == 0:
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        data = readin["surface_type"][
            len1 - index1[-1] - 2 : len1 - index1[0], index2[0] - 1 : index2[-1] + 2
        ]
    elif index2[0] == 0:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        data = readin["surface_type"][
            len1 - index1[-1] - 2 : len1 - index1[0] + 1, index2[0] : index2[-1] + 2
        ]
    else:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        data = readin["surface_type"][
            len1 - index1[-1] - 2 : len1 - index1[0] + 1, index2[0] - 1 : index2[-1] + 2
        ]

    data = np.flipud(data)
    xt_grid, yt_grid = np.meshgrid(xt, yt)
    data_grid = mapping(
        LAT,
        LON,
        data.flatten(),
        yt_grid.flatten(),
        xt_grid.flatten(),
        "nearest",
        np.nan,
    )

    INPUT[:, :, INPUTLIST.index("ast")] = np.copy(data_grid)
    logger.info(f"Surface type processing completed. Unique values: {np.unique(data_grid)}")

    readin.close()
    del [filename, readin, yt, xt, yt_grid, xt_grid, index1, index2, data, data_grid]

    # doy
    logger.info("Processing day of year (doy)")
    data = np.empty(LAT.shape)
    data[:] = int(pd.to_datetime(dd).strftime("%-j"))

    INPUT[:, :, INPUTLIST.index("doy")] = np.copy(data).astype(int)
    logger.debug(f"Day of year set to: {data[0,0]}")
    del data

    # hour
    logger.info("Processing local hour")
    readin = Dataset("./fix/timezones_voronoi_1x1.nc")
    yt = readin["lat"][:]
    xt = readin["lon"][:]
    xt[xt < 0] = xt[xt < 0] + 360
    index1 = np.squeeze(np.argwhere((yt >= floor(lat_lim[0])) & (yt <= ceil(lat_lim[1]))))
    index1 = np.atleast_1d(index1)
    index2 = np.squeeze(np.argwhere((xt >= floor(lon_lim[0])) & (xt <= ceil(lon_lim[1]))))
    index2 = np.atleast_1d(index2)

    offset = np.squeeze(readin["UTC_OFFSET"][0, :, :])

    if (index1[0] == 0) & (index2[0] == 0):
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        offset = offset[index1[0] : index1[-1] + 2, index2[0] : index2[-1] + 2]
    elif index1[0] == 0:
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        offset = offset[index1[0] : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]
    elif index2[0] == 0:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        offset = offset[index1[0] - 1 : index1[-1] + 2, index2[0] : index2[-1] + 2]
    else:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        offset = offset[index1[0] - 1 : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]

    xt_grid, yt_grid = np.meshgrid(xt, yt)
    offset_grid = mapping(
        LAT,
        LON,
        offset.flatten(),
        yt_grid.flatten(),
        xt_grid.flatten(),
        "nearest",
        np.nan,
    )

    data = int(hh) + offset_grid
    data[data < 0] = data[data < 0] + 24
    data[data > 24] = data[data > 24] - 24

    INPUT[:, :, INPUTLIST.index("hour")] = np.copy(data).astype(int)
    logger.info(f"Local hour processing completed. Range: [{data.min()}, {data.max()}]")

    readin.close()
    del [readin, xt, xt_grid, yt, yt_grid, offset, offset_grid, data]

    # fh
    logger.info("Processing forest height (fh)")
    filename = file_finder("fh", dd, initial_hour, forecast_hour)

    if os.path.isfile(filename) is False:
        logger.error(f"Incorrect input file: fh - File not found: {filename}")
        return 1

    logger.debug(f"Reading forest height from: {filename}")
    readin = Dataset(filename)
    yt = readin["lat"][:]
    xt = readin["lon"][:]
    yt = np.round(yt, 3)
    xt = np.round(xt, 3)
    xt[xt < 0] = xt[xt < 0] + 360
    index1 = np.squeeze(np.argwhere((yt >= lat_lim[0]) & (yt <= lat_lim[1])))
    index2 = np.squeeze(np.argwhere((xt >= lon_lim[0]) & (xt <= lon_lim[1])))

    logger.debug(f"Forest height data original shape: {readin['Band1'].shape}")

    # fh, lat x lon
    if (index1[0] == 0) & (index2[0] == 0):
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        data = readin["Band1"][index1[0] : index1[-1] + 2, index2[0] : index2[-1] + 2]
    elif index1[0] == 0:
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        data = readin["Band1"][index1[0] : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]
    elif index2[0] == 0:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        data = readin["Band1"][index1[0] - 1 : index1[-1] + 2, index2[0] : index2[-1] + 2]
    else:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        data = readin["Band1"][index1[0] - 1 : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]

    xt_grid, yt_grid = np.meshgrid(xt, yt)
    data_grid = mapping(
        LAT, LON, data.flatten(), yt_grid.flatten(), xt_grid.flatten(), "linear", 0
    )
    data_grid[data_grid < 0] = 0
    data_grid[np.isnan(data_grid)] = 0

    INPUT[:, :, INPUTLIST.index("fh")] = np.copy(data_grid)
    logger.info(f"Forest height processing completed. Range: [{data_grid.min():.2f}, {data_grid.max():.2f}], "
               f"NaN count: {np.isnan(data_grid).sum()}")

    readin.close()
    del [filename, readin, yt, xt, yt_grid, xt_grid, index1, index2, data, data_grid]

    # vhi
    logger.info("Processing Vegetation Health Index (vhi)")
    filename_npp, filename_j01 = file_finder("vhi", dd, initial_hour, forecast_hour)

    if (os.path.isfile(filename_npp) is False) or (
        os.path.isfile(filename_j01) is False
    ):
        logger.error(f"Incorrect input file: vhi - Files not found: {filename_npp}, {filename_j01}")
        return 1

    # npp
    logger.debug(f"Reading VHI from NPP: {filename_npp}")
    readin = Dataset(filename_npp)
    yt = np.flip(readin["latitude"][:])
    xt = readin["longitude"][:]
    yt = np.round(yt, 3)
    xt = np.round(xt, 3)
    xt[xt < 0] = xt[xt < 0] + 360
    index1 = np.squeeze(np.argwhere((yt >= lat_lim[0]) & (yt <= lat_lim[1])))
    index2 = np.squeeze(np.argwhere((xt >= lon_lim[0]) & (xt <= lon_lim[1])))
    len1 = len(yt)

    if (index1[0] == 0) & (index2[0] == 0):
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        vci_npp = readin["VCI"][
            len1 - index1[-1] - 2 : len1 - index1[0], index2[0] : index2[-1] + 2
        ]
        tci_npp = readin["TCI"][
            len1 - index1[-1] - 2 : len1 - index1[0], index2[0] : index2[-1] + 2
        ]
    elif index1[0] == 0:
        yt = yt[index1[0] : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        vci_npp = readin["VCI"][
            len1 - index1[-1] - 2 : len1 - index1[0], index2[0] - 1 : index2[-1] + 2
        ]
        tci_npp = readin["TCI"][
            len1 - index1[-1] - 2 : len1 - index1[0], index2[0] - 1 : index2[-1] + 2
        ]
    elif index2[0] == 0:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] : index2[-1] + 2]
        vci_npp = readin["VCI"][
            len1 - index1[-1] - 2 : len1 - index1[0] + 1, index2[0] : index2[-1] + 2
        ]
        tci_npp = readin["TCI"][
            len1 - index1[-1] - 2 : len1 - index1[0] + 1, index2[0] : index2[-1] + 2
        ]
    else:
        yt = yt[index1[0] - 1 : index1[-1] + 2]
        xt = xt[index2[0] - 1 : index2[-1] + 2]
        vci_npp = readin["VCI"][
            len1 - index1[-1] - 2 : len1 - index1[0] + 1, index2[0] - 1 : index2[-1] + 2
        ]
        tci_npp = readin["TCI"][
            len1 - index1[-1] - 2 : len1 - index1[0] + 1, index2[0] - 1 : index2[-1] + 2
        ]

    vci_npp = np.flipud(np.asarray(vci_npp))
    tci_npp = np.flipud(np.asarray(tci_npp))
    vci_npp[vci_npp == -999] = np.nan
    tci_npp[tci_npp == -999] = np.nan
    readin.close()
    del readin

    # j01
    logger.debug(f"Reading VHI from J01: {filename_j01}")
    readin = Dataset(filename_j01)

    if (index1[0] == 0) & (index2[0] == 0):
        vci_j01 = readin["VCI"][
            len1 - index1[-1] - 2 : len1 - index1[0], index2[0] : index2[-1] + 2
        ]
        tci_j01 = readin["TCI"][
            len1 - index1[-1] - 2 : len1 - index1[0], index2[0] : index2[-1] + 2
        ]
    elif index1[0] == 0:
        vci_j01 = readin["VCI"][
            len1 - index1[-1] - 2 : len1 - index1[0], index2[0] - 1 : index2[-1] + 2
        ]
        tci_j01 = readin["TCI"][
            len1 - index1[-1] - 2 : len1 - index1[0], index2[0] - 1 : index2[-1] + 2
        ]
    elif index2[0] == 0:
        vci_j01 = readin["VCI"][
            len1 - index1[-1] - 2 : len1 - index1[0] + 1, index2[0] : index2[-1] + 2
        ]
        tci_j01 = readin["TCI"][
            len1 - index1[-1] - 2 : len1 - index1[0] + 1, index2[0] : index2[-1] + 2
        ]
    else:
        vci_j01 = readin["VCI"][
            len1 - index1[-1] - 2 : len1 - index1[0] + 1, index2[0] - 1 : index2[-1] + 2
        ]
        tci_j01 = readin["TCI"][
            len1 - index1[-1] - 2 : len1 - index1[0] + 1, index2[0] - 1 : index2[-1] + 2
        ]

    vci_j01 = np.flipud(np.asarray(vci_j01))
    tci_j01 = np.flipud(np.asarray(tci_j01))
    vci_j01[vci_j01 == -999] = np.nan
    tci_j01[tci_j01 == -999] = np.nan
    readin.close()
    del readin

    # combine two satellites
    logger.debug("Combining NPP and J01 satellite data")
    vci = np.nanmean([vci_npp, vci_j01], axis=0)
    tci = np.nanmean([tci_npp, tci_j01], axis=0)
    vci = vci / 100
    tci = tci / 100
    del [vci_npp, tci_npp, vci_j01, tci_j01]

    # calculate vhi
    data = 0.3 * vci + 0.7 * tci
    data[np.isnan(data)] = -999
    del [vci, tci]

    xt_grid, yt_grid = np.meshgrid(xt, yt)
    data_grid = mapping(
        LAT,
        LON,
        data.flatten(),
        yt_grid.flatten(),
        xt_grid.flatten(),
        "nearest",
        0,
    )
    data_grid[data_grid == -999] = 0

    INPUT[:, :, INPUTLIST.index("vhi")] = np.copy(data_grid)
    logger.info(f"VHI processing completed. Range: [{np.nanmin(data_grid):.3f}, {np.nanmax(data_grid):.3f}], "
               f"NaN count: {np.isnan(data_grid).sum()}")

    del [
        filename_npp,
        filename_j01,
        yt,
        xt,
        yt_grid,
        xt_grid,
        index1,
        index2,
        data,
        data_grid,
    ]

    # t2m
    logger.info("Processing 2m temperature (t2m)")
    filename = file_finder("t2m", dd, initial_hour, forecast_hour)

    if os.path.isfile(filename) is False:
        logger.error(f"Incorrect input file: t2m - File not found: {filename}")
        return 1

    logger.debug(f"Reading temperature from: {filename}")
    readin = xr.open_dataset(
        filename,
        engine="cfgrib",
        filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2},
    )
    yt = readin["latitude"].data
    xt = readin["longitude"].data
    data = readin["t2m"].data
    logger.debug(f"Temperature data shape: {data.shape}, range: [{data.min():.2f}, {data.max():.2f}]")

    data_grid = mapping(
        LAT, LON, data.flatten(), yt.flatten(), xt.flatten(), "linear", np.nan
    )
    data_grid[data_grid < 0] = np.nan

    INPUT[:, :, INPUTLIST.index("t2m")] = np.copy(data_grid)
    logger.info(f"Temperature processing completed. Range: [{np.nanmin(data_grid):.2f}, {np.nanmax(data_grid):.2f}K]")

    readin.close()
    del [filename, readin, yt, xt, data, data_grid]

    # sh2
    logger.info("Processing 2m specific humidity (sh2)")
    filename = file_finder("sh2", dd, initial_hour, forecast_hour)

    if os.path.isfile(filename) is False:
        logger.error(f"Incorrect input file: sh2 - File not found: {filename}")
        return 1

    logger.debug(f"Reading specific humidity from: {filename}")
    readin = xr.open_dataset(
        filename,
        engine="cfgrib",
        filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2},
    )
    yt = readin["latitude"].data
    xt = readin["longitude"].data
    data = readin["sh2"].data
    logger.debug(f"Specific humidity data shape: {data.shape}, range: [{data.min():.6f}, {data.max():.6f}]")

    data_grid = mapping(
        LAT, LON, data.flatten(), yt.flatten(), xt.flatten(), "linear", np.nan
    )
    data_grid[data_grid < 0] = np.nan

    INPUT[:, :, INPUTLIST.index("sh2")] = np.copy(data_grid)
    logger.info(f"Specific humidity processing completed. Range: [{np.nanmin(data_grid):.6f}, {np.nanmax(data_grid):.6f}]")

    readin.close()
    del [filename, readin, yt, xt, data, data_grid]

    # prate
    logger.info("Processing precipitation rate (prate)")
    filename = file_finder("prate", dd, initial_hour, forecast_hour)

    if os.path.isfile(filename) is False:
        logger.error(f"Incorrect input file: prate - File not found: {filename}")
        return 1

    logger.debug(f"Reading precipitation from: {filename}")
    readin = xr.open_dataset(
        filename,
        engine="cfgrib",
        filter_by_keys={"typeOfLevel": "surface", "stepType": "instant"},
    )
    yt = readin["latitude"].data
    xt = readin["longitude"].data
    data = readin["prate"].data
    data = data * 3600  # kg m-2 s-1 -> hourly accumulation
    logger.debug(f"Precipitation data shape: {data.shape}, range: [{data.min():.6f}, {data.max():.6f}]")

    data_grid = mapping(
        LAT, LON, data.flatten(), yt.flatten(), xt.flatten(), "linear", np.nan
    )
    data_grid[data_grid < 0] = np.nan

    INPUT[:, :, INPUTLIST.index("prate")] = np.copy(data_grid)
    logger.info(f"Precipitation processing completed. Range: [{np.nanmin(data_grid):.6f}, {np.nanmax(data_grid):.6f} mm/h]")

    readin.close()
    del [filename, readin, yt, xt, data, data_grid]

    # ws, wd
    logger.info("Processing wind speed and direction (ws, wd)")
    filename = file_finder("ws", dd, initial_hour, forecast_hour)

    if os.path.isfile(filename) is False:
        logger.error(f"Incorrect input file: ws, wd - File not found: {filename}")
        return 1

    logger.debug(f"Reading wind data from: {filename}")
    readin = xr.open_dataset(
        filename,
        engine="cfgrib",
        filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 10},
    )
    yt = readin["latitude"].data
    xt = readin["longitude"].data
    u = readin["u10"].data
    v = readin["v10"].data
    logger.debug(f"Wind components - U shape: {u.shape}, V shape: {v.shape}")

    u_new, v_new = wind_conversion(yt, u, v)

    u_grid = mapping(
        LAT, LON, u_new.flatten(), yt.flatten(), xt.flatten(), "linear", np.nan
    )
    v_grid = mapping(
        LAT, LON, v_new.flatten(), yt.flatten(), xt.flatten(), "linear", np.nan
    )

    # for canopy wind calculation
    UGRID = np.copy(u_grid)
    VGRID = np.copy(v_grid)

    # ws, lat x lon
    ws_grid = np.sqrt(u_grid**2 + v_grid**2)

    # wd, lat x lon
    u_grid = units.Quantity(u_grid, "m/s")
    v_grid = units.Quantity(v_grid, "m/s")
    wd_grid = wind_direction(u_grid, v_grid, convention="from")

    INPUT[:, :, INPUTLIST.index("ws")] = np.copy(ws_grid)
    INPUT[:, :, INPUTLIST.index("wd")] = np.copy(wd_grid)
    logger.info(f"Wind processing completed. Speed range: [{np.nanmin(ws_grid):.2f}, {np.nanmax(ws_grid):.2f} m/s], "
               f"Direction range: [{np.nanmin(wd_grid):.1f}, {np.nanmax(wd_grid):.1f} deg]")

    del [filename, readin, yt, xt, u, v, u_new, v_new, u_grid, v_grid, ws_grid, wd_grid]

    logger.info(f"Start framing... Original data shape: {INPUT.shape}")

    # ---- Fire Framing ----
    XX = LAT.shape[0]
    YY = LAT.shape[1]
    logger.debug(f"Grid dimensions: {XX} x {YY}")

    total = 0
    skip = 0

    # fire mask
    MASK = np.zeros(LAT.shape)
    MASK[FIRE != 0] = 1
    fire_pixels = np.sum(MASK)
    logger.info(f"Fire pixels detected: {fire_pixels}")

    # fire location
    lw, num = ndimage.label(MASK)
    lw = lw.astype(float)
    lw[lw == 0] = np.nan
    logger.info(f"Number of fire clusters: {num}")

    # fire frame
    INPUTFRAME = LATFRAME = LONFRAME = None
    for j in np.arange(1, num + 1, 1):
        total = total + 1
        index = np.argwhere(lw == j)

        if index.shape[0] == 1:
            loc = np.squeeze(index)
        else:
            xx = np.mean(LAT[index[:, 0], index[:, 1]])
            yy = np.mean(LON[index[:, 0], index[:, 1]])
            dis = (LAT - xx) ** 2 + (LON - yy) ** 2
            loc = np.squeeze(np.argwhere(dis == np.min(dis)))
            del [xx, yy, dis]

        if loc.size > 2:
            loc = loc[0, :]

        if (
            (loc[0] - fsize < 0)
            or (loc[0] + fsize + 1 > XX)
            or (loc[1] - fsize < 0)
            or (loc[1] + fsize + 1 > YY)
        ):
            skip = skip + 1
            logger.debug(f"Skipping fire cluster {j} at edge location: {loc}")
            continue

        X_fire = INPUT[
            loc[0] - fsize : loc[0] + fsize + 1, loc[1] - fsize : loc[1] + fsize + 1, :
        ]
        X_lat = LAT[
            loc[0] - fsize : loc[0] + fsize + 1, loc[1] - fsize : loc[1] + fsize + 1
        ]
        X_lon = LON[
            loc[0] - fsize : loc[0] + fsize + 1, loc[1] - fsize : loc[1] + fsize + 1
        ]

        if INPUTFRAME is not None:
            INPUTFRAME = np.append(INPUTFRAME, np.expand_dims(X_fire, axis=0), axis=0)
            LATFRAME = np.append(LATFRAME, np.expand_dims(X_lat, axis=0), axis=0)
            LONFRAME = np.append(LONFRAME, np.expand_dims(X_lon, axis=0), axis=0)
        else:
            INPUTFRAME = np.expand_dims(X_fire, axis=0)
            LATFRAME = np.expand_dims(X_lat, axis=0)
            LONFRAME = np.expand_dims(X_lon, axis=0)

        del [index, loc, X_fire, X_lat, X_lon]
    del [lw, num, INPUT, FIRE, LAT, LON, MASK]

    if "INPUTFRAME" in locals() and INPUTFRAME is not None:
        logger.info(f"Initial frame count: {INPUTFRAME.shape[0]}")
    else:
        logger.warning(f"{tt} no available frames.")
        return 2

    # remove frames with NaN
    index = []
    for i in np.arange(NN):
        X = np.sum(INPUTFRAME[:, :, :, i], axis=(1, 2))
        index = np.append(index, np.squeeze(np.argwhere(np.isnan(X))))
        del X

    index = index.astype(int)

    if index.size != 0:
        logger.error('Removing frames with NaN values...')
        INPUTFRAME = np.delete(INPUTFRAME, index, axis=0)
        LATFRAME = np.delete(LATFRAME, index, axis=0)
        LONFRAME = np.delete(LONFRAME, index, axis=0)
    nan = index.size
    del index

    # remove isolated small fires
    fire_map = np.copy(INPUTFRAME[:, :, :, 0])
    fire_map[fire_map != 0] = 1
    fire_count = np.sum(fire_map, axis=(1, 2))
    fire_max = np.max(INPUTFRAME[:, :, :, 0], axis=(1, 2))
    index = np.squeeze(np.argwhere(fire_count == 1))

    if index.size != 0:
        logger.error('Removing isolated small fires...')
        INPUTFRAME = np.delete(INPUTFRAME, index, axis=0)
        LATFRAME = np.delete(LATFRAME, index, axis=0)
        LONFRAME = np.delete(LONFRAME, index, axis=0)
    small = index.size
    del [fire_map, fire_count, fire_max, index]

    # remove frames with water bodies (AST=17)
    index = np.argwhere(INPUTFRAME[:, :, :, INPUTLIST.index("ast")] == 17)[:, 0]
    index = np.unique(index)

    if index.size != 0:
        logger.error('Removing frames with water bodies...')
        INPUTFRAME = np.delete(INPUTFRAME, index, axis=0)
        LATFRAME = np.delete(LATFRAME, index, axis=0)
        LONFRAME = np.delete(LONFRAME, index, axis=0)
    water = index.size
    del index

    logger.info("Fire framing complete!")

    if INPUTFRAME.shape[0] == 0:
        logger.warning("No available frames after filtering!")
        return 2
    else:
        logger.info(f"Frame statistics - All: {total}, Skip: {skip}, NaN: {nan}, Water: {water}, Small: {small}")
        logger.info(f"Final frame shape: {INPUTFRAME.shape}")

    INPUTFRAME_ori = np.copy(INPUTFRAME)

    # ---- Scaling ----
    logger.info("Applying scaling transformations")
    for X in ["frp", "prate"]:
        i = INPUTLIST.index(X)
        X1 = np.copy(INPUTFRAME[:, :, :, i])
        X1[X1 != 0] = np.log(X1[X1 != 0])
        INPUTFRAME[:, :, :, i] = np.copy(X1)
        logger.debug(f"Applied log scaling to {X}")
        del [i, X1]

    for X in ["sh2", "ws"]:
        i = INPUTLIST.index(X)
        X1 = np.copy(INPUTFRAME[:, :, :, i])
        X1 = np.sqrt(X1)
        INPUTFRAME[:, :, :, i] = np.copy(X1)
        logger.debug(f"Applied sqrt scaling to {X}")
        del [i, X1]

    for X in ["fh", "elv"]:
        i = INPUTLIST.index(X)
        X1 = np.copy(INPUTFRAME[:, :, :, i])
        X1 = (X1) ** (1 / 3)
        INPUTFRAME[:, :, :, i] = np.copy(X1)
        logger.debug(f"Applied cube root scaling to {X}")
        del [i, X1]

    # ---- Normalization ----
    logger.info("Applying normalization")
    # normalization coef
    coef = pd.read_csv("./model/model_normalization_coef.txt")
    coef = np.array(coef)
    a = coef[0, np.append([0, 1], np.arange(4, 15))]
    b = coef[1, np.append([0, 1], np.arange(4, 15))]
    logger.debug(f"Normalization coefficients - a: {a}, b: {b}")

    for i in np.arange(NN):
        X = np.copy(INPUTFRAME[:, :, :, i])
        X = (X - b[i]) / a[i]
        INPUTFRAME[:, :, :, i] = np.copy(X)
        del X
    logger.debug("Normalization completed")

    # ---- Writing Model Input ----
    logger.info(f"Writing model input to: {f_output}")
    INPUTFRAME[np.isnan(INPUTFRAME)] = -999

    f = Dataset(f_output, "w")
    f.createDimension("time", 1)
    f.createDimension("flen", INPUTFRAME.shape[0])
    f.createDimension("xlen", INPUTFRAME.shape[1])
    f.createDimension("ylen", INPUTFRAME.shape[2])
    f.createDimension("num_input", NN)
    var_time = f.createVariable("time", str, ("time",))
    var_input0 = f.createVariable(
        "input_noscale", "float", ("flen", "xlen", "ylen", "num_input")
    )
    var_input = f.createVariable(
        "input", "float", ("flen", "xlen", "ylen", "num_input")
    )
    var_lat = f.createVariable("frame_lat", "float", ("flen", "xlen", "ylen"))
    var_lon = f.createVariable("frame_lon", "float", ("flen", "xlen", "ylen"))
    var_list = f.createVariable("INPUTLIST", str, ("num_input",))

    var_time[:] = np.array(time).astype(str)
    var_input0[:] = INPUTFRAME_ori
    var_input[:] = INPUTFRAME
    var_lat[:] = LATFRAME
    var_lon[:] = LONFRAME
    var_list[:] = np.array(INPUTLIST).astype(str)
    f.close()
    
    logger.info(f"Successfully wrote output file with {INPUTFRAME.shape[0]} frames")
    logger.info("main_driver completed successfully")
    return 0
