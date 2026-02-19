import argparse
from netCDF4 import Dataset
import dask.array as da
import zarr
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
import glob

def convert_sample_to_npy(viirs_folders, i, folder):
    X_folder = np.zeros((12, 135, 135, 13))
    try:
        for j in range(12):
            hour = str(int(folder[-2:]) + j).zfill(2)
            if not os.path.exists(folder[:-2] + hour):
                raise Exception("File does not exist")
            file = folder[:-2] + hour + '/perim*'
            file = glob.glob(file)[0]
            with Dataset(file, 'r') as data:
                X_folder[j] = data.variables['input'][:].squeeze()
    except Exception as e:
        return None, None

    file = viirs_folders[i] + '/perim*'
    file = glob.glob(file)[0]
    with Dataset(file, 'r') as data:
        LAT = data.variables['frame_lat'][:].squeeze()[:, 0]
        LON = data.variables['frame_lon'][:].squeeze()[0, :]

    file = viirs_folders[i + 1] + '/viirs.perim.conus.out*'
    file = glob.glob(file)[0]
    with Dataset(file, 'r') as data:
        yt = data.variables['grid_lat'][:, 0]
        xt = data.variables['grid_lon'][0, :]
        index1 = np.squeeze(np.argwhere((yt >= LAT[0]) & (yt <= LAT[-1])))
        index2 = np.squeeze(np.argwhere((xt >= LON[0]) & (xt <= LON[-1])))
        y_data = data.variables['fire'][:, index1[0]:index1[-1]+1, index2[0]:index2[-1]+1].squeeze()
        y_folder = y_data.filled(0).copy()
        y_folder.resize(135, 135)

    return X_folder, y_folder

def convert_to_npy(output_folder):
    X = []
    y = []

    for folder in glob.glob('./input/*'):
        if not os.path.isdir(folder):
            continue

        viirs_folders = (
            glob.glob(folder + '/*00')
            + glob.glob(folder + '/*12')
        )
        viirs_folders.sort()

        X_fire_list = []
        y_fire_list = []
        for i, folder in enumerate(viirs_folders[:-1]):
            print(folder)
            X_folder, y_folder = convert_sample_to_npy(
                viirs_folders,
                i,
                folder
            )
            if X_folder is None:
                continue
            X_fire_list.append(X_folder)
            y_fire_list.append(y_folder)

        if len(X_fire_list) == 0:
            continue

        parent_folder = os.path.basename(Path(folder).parent)

        if not os.path.exists(os.path.join('tmp', '{}'.format(parent_folder))):
            os.makedirs('tmp/{}'.format(parent_folder))

        path = os.path.join('tmp', '{}'.format(parent_folder))
        X_shape = (len(X_fire_list),) + X_fire_list[0].shape
        X_fire = zarr.open(
            path + '/X.zarr',
            mode='a',
            shape=X_shape,
            chunks=(1,) + X_shape[1:],
            dtype='float32'
        )
        y_shape = (len(y_fire_list),) + y_fire_list[0].shape
        y_fire = zarr.open(
            path + '/y.zarr',
            mode='a',
            shape=y_shape,
            chunks=(1,) + y_shape[1:],
            dtype='float32'
        )
        X_fire[:] = np.stack(X_fire_list, axis=0)
        y_fire[:] = np.stack(y_fire_list, axis=0)

        X.append(path + '/X.zarr')
        y.append(path + '/y.zarr')

    X_da = da.concatenate([da.from_zarr(x) for x in X], axis=0)
    y_da = da.concatenate([da.from_zarr(y) for y in y], axis=0)
    X_da.to_zarr(os.path.join(output_folder, 'X.zarr'))
    y_da.to_zarr(os.path.join(output_folder, 'y.zarr'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", default='.', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.destination):
        os.makedirs(args.destination)

    convert_to_npy(args.destination)
