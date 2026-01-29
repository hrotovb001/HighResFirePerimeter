import os
import argparse
import sys
import re
from datetime import datetime

def parse_filename(filename):
    # Extract satellite: npp or j01
    sat_pattern = r'(npp|j01)'
    sat_match = re.search(sat_pattern, filename, re.IGNORECASE)
    satellite = sat_match.group(1).lower() if sat_match else None

    # Extract start date: sYYYYMMDD
    date_pattern = r's(\d{8})'
    date_match = re.search(date_pattern, filename)

    if not satellite or not date_match:
        return None, None

    try:
        start_str = date_match.group(1)
        start_date = datetime.strptime(start_str, '%Y%m%d')
        year = start_date.strftime('%Y')
        iso_week = f"{start_date.isocalendar()[1]:02d}"
        pcode = f"P{year}0{iso_week}"
        return satellite, pcode
    except ValueError:
        return None, None

def rename_files(directory, dry_run=False):
    renamed_count = 0

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if not filename.endswith('.nc'):
                continue

            old_path = os.path.join(root, filename)
            satellite, pcode = parse_filename(filename)

            if not satellite or not pcode:
                print(f"Skipping {filename}: cannot parse satellite or date", file=sys.stderr)
                continue

            new_filename = f"VHP.G04.C07.{satellite}.{pcode}.VH.nc"
            
            # Create satellite subdirectory
            new_root = os.path.join(root, satellite)
            os.makedirs(new_root, exist_ok=True)
            new_path = os.path.join(new_root, new_filename)
            
            if dry_run:
                print(f"Would rename: {old_path}")
                print(f"          to: {new_path}")
                print()
            else:
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                    renamed_count += 1
                except OSError as e:
                    print(f"Error renaming {old_path}: {e}", file=sys.stderr)
        break

    return renamed_count

def main():
    parser = argparse.ArgumentParser(description="Rename GVH NetCDF files to VHP format")
    parser.add_argument("directory", help="Directory containing GVH files")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes only")

    args = parser.parse_args()

    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist", file=sys.stderr)
        sys.exit(1)

    count = rename_files(args.directory, args.dry_run)

    if not args.dry_run:
        print(f"\nRenamed {count} files total.")

if __name__ == "__main__":
    main()

