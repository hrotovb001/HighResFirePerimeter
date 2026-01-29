from google.cloud import storage
from datetime import datetime
import argparse
import pandas as pd
import os

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded storage object {source_blob_name} from bucket {bucket_name} to local file {destination_file_name}.")

bucket_name = "high-resolution-rapid-refresh"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str)
    parser.add_argument("--end", type=str)
    parser.add_argument("--destination", type=str)
    args = parser.parse_args()

    start_time = datetime.strptime(args.start, "%Y%m%d%H%M")
    end_time = datetime.strptime(args.end, "%Y%m%d%H%M")
    for time in pd.date_range(start_time, end_time, freq="1h"):
        source_blob_name = (
            "hrrr."
            + time.strftime("%Y%m%d")
            + "/conus/hrrr.t"
            + time.strftime("%H")
            + "z.wrfsfcf00.grib2"
        )
        destination_file_name = (
            args.destination
            + "/"
            + time.strftime("%Y%m%d")
            + "/hrrr.t"
            + time.strftime("%H")
            + "z.wrfsfcf00.grib2"
        )
        if not os.path.exists(os.path.dirname(destination_file_name)):
            os.makedirs(os.path.dirname(destination_file_name))
        download_blob(bucket_name, source_blob_name, destination_file_name)
