import argparse
import os
from pathlib import Path
import geopandas as gpd
from geocube.api.core import make_geocube

def main():
    parser = argparse.ArgumentParser(
        description="Rasterize GeoJSON features to NetCDF files"
    )
    parser.add_argument(
        "-i", "--input-dir",
        required=True,
        help="Input directory containing .geojson files"
    )
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Output directory for .nc files"
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {input_path}")
    
    # Find all .geojson files
    geojson_files = list(input_path.glob("*.geojson"))
    if not geojson_files:
        print("No .geojson files found in input directory")
        return
    
    print(f"Found {len(geojson_files)} GeoJSON file(s)")
    
    resolution = (-0.001, 0.001)
    
    for geojson_file in geojson_files:
        print(f"Processing {geojson_file.name}...")
        gdf = gpd.read_file(geojson_file)
        gdf = gdf.sort_values(by=['primarykey'], ascending=False)
        gdf['fire'] = 1
        
        for row in gdf.itertuples():
            out_grid = make_geocube(
                vector_data=gdf.iloc[[row.Index]],
                measurements=['fire'],
                fill=0,
                resolution=resolution,
            )
            output_file = output_path / f"{row.primarykey}.nc"
            out_grid.to_netcdf(output_file)
            print(f"  Created {output_file.name}")

if __name__ == "__main__":
    main()

