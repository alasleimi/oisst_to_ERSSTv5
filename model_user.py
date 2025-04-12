#!/usr/bin/env python
"""
Prediction Script

This script:
  1. Loads the NOAA target data (sst.mnmean.nc) to get the lat/lon grid.
  2. Loads and “cleans” the new input dataset (oisst_new.nc) exactly as done
     for oisst_monthly.nc during training.
  3. Loads the saved models (from saved_models/) whose filenames encode the target
     pixel (i, j).
  4. For each grid cell, it uses the corresponding model’s saved patch size and
     feature mask to extract the identical patch from the new input data, applies
     any scaler (if present), and predicts the SST value.
  5. Finally, it appends the new predicted month into the NOAA file using NCO.
  
Make sure that all required packages (xarray, joblib, netCDF4, subprocess, etc.) are installed.
"""

import argparse
import os
import sys
import math
import random
import joblib
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import subprocess
from netCDF4 import Dataset, date2num

MONTH_STRING = "2025-03-01"
# =============================================================================
# Custom Model Classes (from training) -- must be present for unpickling
# =============================================================================

from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, OrthogonalMatchingPursuitCV
from sklearn.ensemble import RandomForestRegressor

class XGBRidgeCVRegressor(BaseEstimator, RegressorMixin):
    """
    Combined model that uses XGBRegressor for feature selection and then
    fits a Ridge on the selected features.
    """
    def __init__(self, importance_threshold=None, **xgb_params):
        self.importance_threshold = importance_threshold
        self.xgb_params = xgb_params
        if 'n_estimators' not in self.xgb_params:
            self.xgb_params['n_estimators'] = 100
        self.selector = None
        self.ridge = None
        self.feature_mask_ = None

    def fit(self, X, y):
        self.selector = XGBRegressor(**self.xgb_params)
        self.selector.fit(X, y)
        importances = self.selector.feature_importances_
        threshold = self.importance_threshold if self.importance_threshold is not None else np.median(importances)
        self.feature_mask_ = importances >= threshold
        if not np.any(self.feature_mask_):
            self.feature_mask_ = np.ones(X.shape[1], dtype=bool)
        self.ridge = Ridge()
        self.ridge.fit(X[:, self.feature_mask_], y)
        return self

    def predict(self, X):
        if self.feature_mask_ is None:
            raise ValueError("The model has not been fitted yet.")
        return self.ridge.predict(X[:, self.feature_mask_])


class RFQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    Custom estimator that wraps a RandomForestRegressor to perform quantile regression.
    """
    def __init__(self, quantile=0.5, **rf_kwargs):
        self.quantile = quantile
        self.rf_kwargs = rf_kwargs
        self.rf = None

    def fit(self, X, y):
        self.rf = RandomForestRegressor(**self.rf_kwargs)
        self.rf.fit(X, y)
        return self

    def predict(self, X):
        all_preds = np.stack([estimator.predict(X) for estimator in self.rf.estimators_], axis=1)
        return np.percentile(all_preds, self.quantile * 100, axis=1)


# =============================================================================
# GeoProcessor (for patch extraction)
# =============================================================================
class GeoProcessor:
    def __init__(self, lats, lons):
        self.lats = lats
        self.lons = lons
        self.lon_domain = "0to360" if lons.min() >= 0 else "-180to180"

    def wrap_longitude(self, lon):
        if self.lon_domain == "-180to180":
            return ((lon + 180) % 360) - 180
        return lon % 360

    def get_nearest_indices(self, target_lat, target_lon, n):
        # Select n closest lat indices
        lat_diff = np.abs(self.lats - target_lat)
        lat_idx_unsorted = np.argpartition(lat_diff, n - 1)[:n]
        lat_idx = np.sort(lat_idx_unsorted)

        # Select n closest lon indices
        target_lon = self.wrap_longitude(target_lon)
        lon_diff = np.abs(self.lons - target_lon)
        lon_diff = np.minimum(lon_diff, 360 - lon_diff)
        lon_idx_unsorted = np.argpartition(lon_diff, n - 1)[:n]
        lon_idx = np.sort(lon_idx_unsorted)

        return lat_idx, lon_idx


# =============================================================================
# Data Loading and Cleaning (exactly as in training)
# =============================================================================
def load_and_preprocess_data(target_path, input_path):
    """
    Loads the target (NOAA ERSV5) and input (NOAA OISST) datasets,
    aligns time coordinates (to month start), selects common time range,
    cleans missing values, loads data into memory, and plots the last month.
    """
    import xarray as xr
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from dask.diagnostics import ProgressBar

    # Open datasets with dask chunks along time.
    target_ds = xr.open_dataset(target_path, decode_times=True, chunks={"time": 1})
    input_ds = xr.open_dataset(input_path, decode_times=True, chunks={"time": 1})
    
    def shift_time_to_month_start(time_dataarray):
        times = pd.to_datetime(time_dataarray.values)
        return times.to_period("M").to_timestamp()
    
    target_ds = target_ds.assign_coords(time=("time", shift_time_to_month_start(target_ds["time"])))
    input_ds = input_ds.assign_coords(time=("time", shift_time_to_month_start(input_ds["time"])))
    

    
    # Process target sst
    raw_target = target_ds["sst"].values
    print("Raw target sst: min = {}, max = {}".format(np.nanmin(raw_target), np.nanmax(raw_target)))
    NOAA_MISSING = -9.96921e+36
    target_sst = target_ds["sst"].where(np.isfinite(target_ds["sst"]), np.nan)
    target_sst = target_sst.where(target_sst != NOAA_MISSING, np.nan)
    clean_target = target_sst.values
    print("Cleaned target sst: min = {}, max = {}".format(np.nanmin(clean_target), np.nanmax(clean_target)))
    
    # Process input sst
    raw_input = input_ds["sst"].values
    print("Raw input sst: min = {}, max = {}".format(np.nanmin(raw_input), np.nanmax(raw_input)))
    input_sst = input_ds["sst"].where(np.isfinite(input_ds["sst"]), np.nan)
    input_sst = input_sst.where(input_sst != -999, np.nan)
    clean_input = input_sst.values
    print("Cleaned input sst: min = {}, max = {}".format(np.nanmin(clean_input), np.nanmax(clean_input)))
    
    with ProgressBar():
        target_data = target_sst.load()
        input_data = input_sst.load()
    
    # -------------------- DEBUGGING CODE (optional) --------------------
    before_last_time = target_data["time"].values[-2]
    target_val_00 = target_data.sel(time=before_last_time, lat=0, lon=0, method="nearest")
    input_val_00 = input_data.sel(time=before_last_time, lat=0, lon=0, method="nearest")
    print(f"DEBUG: Target (0,0) value for before last month ({before_last_time}): {target_val_00.values}")
    print(f"DEBUG: Input (0,0) value for before last month ({before_last_time}): {input_val_00.values}")
    target_00_all = target_data.sel(lat=0, lon=0, method="nearest")
    nan_mask_target = np.isnan(target_00_all.values)
    if np.any(nan_mask_target):
        print("DEBUG: Target grid cell (0,0) is NaN in the following months:")
        for t in target_00_all.time.values[nan_mask_target]:
            print(pd.to_datetime(t).strftime("%Y-%m"))
    else:
        print("DEBUG: Target grid cell (0,0) is never NaN in any month.")
    input_00_all = input_data.sel(lat=0, lon=0, method="nearest")
    nan_mask_input = np.isnan(input_00_all.values)
    if np.any(nan_mask_input):
        print("DEBUG: Input grid cell (0,0) is NaN in the following months:")
        for t in input_00_all.time.values[nan_mask_input]:
            print(pd.to_datetime(t).strftime("%Y-%m"))
    else:
        print("DEBUG: Input grid cell (0,0) is never NaN in any month.")
    # ------------------ END DEBUGGING CODE ----------------------
    
    
    return target_data, input_data


# =============================================================================
# NCO Functions for Appending the New Month
# =============================================================================
def nco_add_one_month(
    original_nc: str,
    pred_data: np.ndarray,
    dec_time_str: str,
    out_nc: str = "sst_extended.nc"
):
    """
    Use NCO (ncrcat) to append one new monthly time slice to 'original_nc',
    containing variable "sst" (and possibly "time_bnds").
    Produces 'out_nc' as the final appended file.
    """
    new_month_nc = "new_month.nc"
    dec_time_dt = datetime.datetime.strptime(dec_time_str, "%Y-%m-%d")

    with Dataset(original_nc, "r") as ds_src:
        lat = ds_src.variables["lat"][:]
        lon = ds_src.variables["lon"][:]

        time_var = ds_src.variables["time"]
        time_units    = getattr(time_var, "units", "days since 1800-01-01")
        time_calendar = getattr(time_var, "calendar", "gregorian")

        has_time_bnds = ("time_bnds" in ds_src.variables)
        if has_time_bnds:
            tb_var = ds_src.variables["time_bnds"]
            tb_attrs = dict(tb_var.__dict__)

    time_val = date2num(dec_time_dt, time_units, time_calendar)

    next_month_dt = dec_time_dt.replace(day=1) + datetime.timedelta(days=32)
    next_month_dt = next_month_dt.replace(day=1)
    t_bnds_vals = [
        time_val,
        date2num(next_month_dt, time_units, time_calendar)
    ]

    with Dataset(new_month_nc, "w", format="NETCDF4") as ds_out:
        ds_out.createDimension("time", 1)
        ds_out.createDimension("lat", len(lat))
        ds_out.createDimension("lon", len(lon))
        if has_time_bnds:
            ds_out.createDimension("nbnds", 2)

        tvar = ds_out.createVariable("time", "f4", ("time",))
        latvar = ds_out.createVariable("lat",  "f4", ("lat",))
        lonvar = ds_out.createVariable("lon",  "f4", ("lon",))
        sstvar = ds_out.createVariable("sst",  "f4", ("time","lat","lon",))

        tvar[:]  = [time_val]
        latvar[:] = lat
        lonvar[:] = lon

        if has_time_bnds:
            tbvar = ds_out.createVariable("time_bnds", "f4", ("time","nbnds"))
            tbvar[0, :] = t_bnds_vals
            for attr_name, attr_val in tb_attrs.items():
                setattr(tbvar, attr_name, attr_val)

        if pred_data.shape != (len(lat), len(lon)):
            raise ValueError(
                f"pred_data.shape={pred_data.shape} != (lat, lon)=({len(lat)}, {len(lon)})"
            )
        sstvar[0,:,:] = pred_data

        tvar.units = time_units
        tvar.calendar = time_calendar
        latvar.units = "degrees_north"
        lonvar.units = "degrees_east"

    cmd_concat = ["ncrcat", original_nc, new_month_nc, out_nc]
    subprocess.run(cmd_concat, check=True)

    for f in [ new_month_nc]:
        if os.path.exists(f):
            os.remove(f)

    print(f"Appended new time {dec_time_str} into {out_nc} successfully.")


def save_noaa_with_prediction(noaa_path, pred_data, dec_time):
    """
    A thin wrapper to call nco_add_one_month.
    """
    nco_add_one_month(
        original_nc=noaa_path,
        pred_data=pred_data,
        dec_time_str=dec_time,
        out_nc="sst_extended.nc"
    )


# =============================================================================
# Main Prediction Routine
# =============================================================================
def main():
    MODEL_DIR = "saved_models"  # Directory where models were saved during training
    # In prediction we use the NOAA target file for the grid and the new input file for features.
    target_path = "sst.mnmean.nc"   # Use the same target as in training (for grid coordinates)
    new_input_path = "oisst_new.nc"   # New input file (last month is the “X” for prediction)
    print("Loading data ...")
    target_data, new_input_data = load_and_preprocess_data(target_path, new_input_path)
    
    # Create a GeoProcessor for the new input dataset (for patch extraction)
    new_input_geo = GeoProcessor(new_input_data.lat.values, new_input_data.lon.values)
    
    nlat = len(target_data.lat)
    nlon = len(target_data.lon)
    # Create an empty array for the predictions (shape: [nlat, nlon])
    predictions = np.full((nlat, nlon), np.nan, dtype=np.float32)
    
    # Load saved model files (each filename is assumed to be "i_j_metric.joblib")
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")]
    models_dict = {}  # mapping: (i, j) -> model_dict
    for f in tqdm(model_files, desc="loading models"):
        try:
            parts = f[:-7].split("_")
            if len(parts) < 3:
                print("Skipping file with unexpected format:", f)
                continue
            i = int(parts[0])
            j = int(parts[1])
            # If multiple models exist for one pixel, choose the one with higher R2 (if available)
            current = models_dict.get((i, j))
            model_dict = joblib.load(os.path.join(MODEL_DIR, f))
            if current is None:
                models_dict[(i, j)] = model_dict
            else:
                if model_dict.get("r2", -np.inf) > current.get("r2", -np.inf):
                    models_dict[(i, j)] = model_dict
        except Exception as e:
            print(f"Error loading model file {f}: {e}")
    
    print(f"Found models for {len(models_dict)} pixels out of {nlat*nlon} total.")

    # Loop over every grid point in the target file and predict SST if a model exists.
    for i in tqdm(range(nlat), desc="Predicting over grid"):
        lat_val = target_data.lat.values[i]
        for j in range(nlon):
            lon_val = target_data.lon.values[j]
            # Only predict if a model was trained for this pixel.
            if (i, j) in models_dict:
                model_info = models_dict[(i, j)]
                patch_size = model_info.get("patch_size", 32)  # Use saved patch_size (default 32)
                # Replicate the patch extraction exactly as during training:
                lat_idx, lon_idx = new_input_geo.get_nearest_indices(lat_val, lon_val, patch_size)
                # Extract the patch from new input data.
                patch = new_input_data.isel(lat=lat_idx, lon=lon_idx)
                if "time" in patch.dims:
                    patch = patch.isel(time=0)  # For the new month, there is only one time step.
                X_patch = patch.values.reshape(1, -1)  # Flatten spatial dimensions
                
                # Use the saved feature mask (computed during training) to select columns.
                feature_mask = model_info.get("feature_mask", np.ones(X_patch.shape[1], dtype=bool))
                if feature_mask.shape[0] != X_patch.shape[1]:
                    print(f"Warning: Feature mask size {feature_mask.shape[0]} does not match patch size {X_patch.shape[1]} for pixel ({i},{j}). Skipping prediction for this pixel.")
                    continue
                X_patch_filtered = X_patch[:, feature_mask]
                
                # If there are NaNs in the new features, set prediction to NaN.
                if np.isnan(X_patch_filtered).any():
                    predictions[i, j] = np.nan
                else:
                    # If a scaler was saved during training, apply it.
                    scaler = model_info.get("scaler", None)
                    if scaler is not None:
                        try:
                            X_patch_filtered = scaler.transform(X_patch_filtered)
                        except Exception as e:
                            print(f"Scaler transform error at pixel ({i},{j}): {e}")
                    model = model_info.get("model", None)
                    if model is None:
                        print(f"No model found in model_info for pixel ({i},{j}).")
                        continue
                    try:
                        pred = model.predict(X_patch_filtered)
                        predictions[i, j] = pred[0]  # prediction is a 1-element array
                    except Exception as e:
                        print(f"Prediction error at pixel ({i},{j}): {e}")
                        predictions[i, j] = np.nan
            # If no model exists for (i,j), leave the prediction as NaN.
    
    # (Optional) Plot the prediction field.
    if False:
        plt.figure(figsize=(8, 6))
        plt.imshow(predictions, origin='lower')
        plt.title("Predicted SST for New Month")
        plt.colorbar(label="Predicted SST")
        plt.show()
    
    # Save the prediction by appending the new month into the NOAA file.
    # (Here dec_time is set to "2025-01-01"; change as appropriate.)
    save_noaa_with_prediction(target_path, predictions, MONTH_STRING)

def parse_args():
    """
    Parse command-line arguments for the script.
    Returns:
        str: The month string provided by the user, if any.
    """
    parser = argparse.ArgumentParser(description="Process a specific month string.")
    parser.add_argument(
        "--month",
        type=str,
        help="Month string in YYYY-MM-DD format (e.g., 2025-04-01)",
        default=datetime.date.today().strftime("%Y-%m-%d")
    )
    args = parser.parse_args()


    try:
        datetime.datetime.strptime(args.month, "%Y-%m-%d")
    except ValueError:
        parser.error("The --month argument must be in YYYY-MM-DD format.")
    
    global MONTH_STRING
    MONTH_STRING = args.month
     
    
if __name__ == "__main__":
    parse_args()
    main()
