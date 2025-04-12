#!/usr/bin/env python3
"""
Download NOAA OISST V2.1 daily data.

This script downloads NOAA OISST V2.1 NetCDF files using the following URL pattern:
  https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/{year}{month}/oisst-avhrr-v02r01.{year}{month}{day}.nc

It supports two modes:
  - sample mode: downloads data for February 2022 (28 days)
  - full mode: downloads data for a specified date range; if not provided, defaults to 1981-01-01 to 2024-12-31

Each day's file is saved as "oisst-avhrr-v02r01.YYYYMMDD.nc" in the output folder.
"""

import requests
import os
import argparse
from datetime import datetime, timedelta

def download_day(date_obj, output_folder):
    """
    Download NOAA OISST data for a single day and save as oisst-avhrr-v02r01.YYYYMMDD.nc.
    The URL is constructed as:
      https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/{year}{month}/oisst-avhrr-v02r01.{year}{month}{day}.nc
    """
    year_str = date_obj.strftime("%Y")
    month_str = date_obj.strftime("%m")
    day_str = date_obj.strftime("%d")
    
    # Construct the URL using the year-month folder and filename pattern.
    url = (f"https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/"
           f"access/avhrr/{year_str}{month_str}/oisst-avhrr-v02r01.{year_str}{month_str}{day_str}.nc")
    
    filename = os.path.join(output_folder, f"oisst-avhrr-v02r01.{year_str}{month_str}{day_str}.nc")
    
    if os.path.exists(filename):
        print(f"File already exists: {filename}. Skipping download.")
        return
    
    print(f"Downloading NOAA OISST data for {year_str}-{month_str}-{day_str} ...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded and saved: {filename}")
    else:
        print(f"Failed to download {url}. HTTP status code: {response.status_code}")

def generate_date_range(start_date_str, end_date_str):
    """Generate a list of date objects from start_date to end_date inclusive."""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    delta = end_date - start_date
    return [start_date + timedelta(days=i) for i in range(delta.days + 1)]

def main():
    parser = argparse.ArgumentParser(description="Download NOAA OISST V2.1 daily data (one file per day).")
    parser.add_argument("--mode", type=str, choices=["sample", "full"], required=True,
                        help="Download mode: 'sample' downloads Feb 2022 (28 days); 'full' downloads a date range.")
    parser.add_argument("--output_folder", type=str, 
                        help="Folder to save the downloaded files.", default="noaa_sst")
    parser.add_argument("--start_date", type=str, default=None,
                        help="(full mode only) Start date in YYYY-MM-DD format. Defaults to 1981-01-01 if not provided.")
    parser.add_argument("--end_date", type=str, default=None,
                        help="(full mode only) End date in YYYY-MM-DD format. Defaults to 2024-12-31 if not provided.")
    args = parser.parse_args()
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    if args.mode == "sample":
        # Sample mode: download data for February 2022 (28 days)
        start_date_str = "2022-02-01"
        end_date_str = "2022-02-28"
    else:
        # Full mode: use provided dates or default to 1981-01-01 to 2024-12-31.
        start_date_str = args.start_date if args.start_date is not None else "1981-01-01"
        end_date_str = args.end_date if args.end_date is not None else "2024-12-31"
    
    date_list = generate_date_range(start_date_str, end_date_str)
    total_days = len(date_list)
    print(f"Downloading NOAA OISST data for {total_days} days (from {start_date_str} to {end_date_str}).")
    
    for date_obj in date_list:
        download_day(date_obj, args.output_folder)

if __name__ == "__main__":
    main()
