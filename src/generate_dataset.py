import geopandas as gpd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset for HighResFirePerimeter"
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


