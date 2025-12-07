import geopandas as gpd
from pathlib import Path
import perim_preprocessor
import perimeter_inputgen

def main():
    namelist = pd.read_csv("./input/namelist", header=None, delimiter="=")
    namelist = namelist[1]

    for file in os.listdir('fires'):
        gdf = gpd.read_file(os.path.join('fires', file))
        gdf = gdf.sort_values(by='primarykey', ascending=True)
        for _, perimeter in gdf.iterrows():
            if perimeter["primarykey"].split("|")[0] == "CONUS" and perimeter["primarykey"].split("|")[2] > "2025-06-19T00:00:00" and perimeter["primarykey"].split("|")[2] < "2025-07-19T00:00:00":
                digits_only_time = int(''.join(ch for ch in perimeter["primarykey"].split("|")[2] if ch.isdigit())[:10])
                lat_lim = gdf.geometry.bounds[2:4]
                lon_lim = gdf.geometry.bounds[0:2]
                if lat_lim[1] - lat_lim[0] < 0.3:
                    lat_lim[0] -= 0.3 - (lat_lim[1] - lat_lim[0])
                    lat_lim[1] += 0.3 - (lat_lim[1] - lat_lim[0])
                if lon_lim[1] - lon_lim[0] < 0.3:
                    lon_lim[0] -= 0.3 - (lon_lim[1] - lon_lim[0])
                    lon_lim[1] += 0.3 - (lon_lim[1] - lon_lim[0])
                preprocessor(namelist[0], perimeter["fireid"], digits_only_time, lat_lim, lon_lim)
                for i in range(12):
                    main_driver(digits_only_time[8:] + i, 0, "./input/" + digits_only_time + "/" + namelist[0] + "." + digits_only_time + ".nc", "./input/" + digits_only_time + "/" + namelist[1] + "." + digits_only_time + ".nc", lat_lim, lon_lim)
