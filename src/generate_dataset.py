import geopandas as gpd
from pathlib import Path

def main():
    namelist = pd.read_csv("./input/namelist", header=None, delimiter="=")
    namelist = namelist[1]

    # TODO: Implement dataset generation using perim preprocessor and perimeter inputgen

