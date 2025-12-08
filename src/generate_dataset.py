import geopandas as gpd
from pathlib import Path
import perim_preprocessor
import perimeter_inputgen
import argparse
import os
import pandas as pd
from datetime import datetime, timedelta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start_date", help="Start date in YYYY-MM-DDTHH:MM:SS format")
    parser.add_argument("end_date", help="End date in YYYY-MM-DDTHH:MM:SS format")
    args = parser.parse_args()

    namelist = pd.read_csv("./input/namelist", header=None, delimiter="=")
    namelist = namelist[1]

    for file in os.listdir('fires'):
        gdf = gpd.read_file(os.path.join('fires', file))
        gdf = gdf.sort_values(by='primarykey', ascending=True)
        for _, perimeter in gdf.iterrows():
            if perimeter["primarykey"].split("|")[0] != "CONUS" or perimeter["primarykey"].split("|")[2] < args.start_date or perimeter["primarykey"].split("|")[2] > args.end_date:
                break

            time = datetime.fromisoformat(perimeter["primarykey"].split("|")[2])

            bounds = perimeter.geometry.bounds
            lat_lim = [bounds[1], bounds[3]]
            lon_lim = [bounds[0], bounds[2]]
            if lon_lim[0] < 0:
                lon_lim[0] += 360
            if lon_lim[1] < 0:
                lon_lim[1] += 360

            y_offset = (0.2 - lat_lim[1] + lat_lim[0]) / 2
            x_offset = (0.2 - lon_lim[1] + lon_lim[0]) / 2
            lat_lim[0] -= y_offset
            lat_lim[1] += y_offset
            lon_lim[0] -= x_offset
            lon_lim[1] += x_offset

            if not os.path.exists("./input/" + str(perimeter["fireid"]) + "/" + time.strftime("%Y%m%d%H")):
                os.makedirs("./input/" + str(perimeter["fireid"]) + "/" + time.strftime("%Y%m%d%H"))
            perim_preprocessor.preprocessor(namelist[0][1:], str(perimeter["fireid"]), time.strftime("%Y%m%d%H"), lat_lim, lon_lim)
            try:
                if not os.path.exists("./input/" + str(perimeter["fireid"]) + "/" + (time + timedelta(hours=12)).strftime("%Y%m%d%H")):
                    os.makedirs("./input/" + str(perimeter["fireid"]) + "/" + (time + timedelta(hours=12)).strftime("%Y%m%d%H"))
                perim_preprocessor.preprocessor(namelist[0][1:], str(perimeter["fireid"]), (time + timedelta(hours=12)).strftime("%Y%m%d%H"), lat_lim, lon_lim)
            except:
                break
            for i in range(12):
                current_time = time + timedelta(hours=i)
                current_time_str = current_time.strftime("%Y%m%d%H")
                if not os.path.exists("./input/" + str(perimeter["fireid"]) + "/" + current_time_str):
                    os.makedirs("./input/" + str(perimeter["fireid"]) + "/" + current_time_str)
                f_input = "./input/" + str(perimeter["fireid"]) + "/" + time.strftime("%Y%m%d%H") + "/" + namelist[0][1:] + "." + time.strftime("%Y%m%d%H") + ".nc"
                f_output = "./input/" + str(perimeter["fireid"]) + "/" + current_time_str + "/" + namelist[1][1:] + "." + current_time_str + ".nc"
                result = perimeter_inputgen.main_driver(current_time.hour, 0, f_input, f_output, lat_lim, lon_lim)
                if result != 0:
                    break

if __name__ == "__main__":
    main()
