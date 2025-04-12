import math
import os
import random
import joblib
import numpy as np
from multiprocessing import Process, Queue
import time
import glob
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.externals._packaging.version import Infinity
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import argparse
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar

# Linear models and other regressors:
from sklearn.linear_model import (
    ElasticNetCV, OrthogonalMatchingPursuitCV, Ridge, Lasso, ElasticNet, LinearRegression, HuberRegressor,
    BayesianRidge, RidgeCV, TheilSenRegressor, RANSACRegressor, QuantileRegressor,
    Lars, LassoLars, OrthogonalMatchingPursuit, ARDRegression
)
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# For parallel scheduling
import concurrent.futures
import threading

# Global settings
RANDOM_SEED = 42
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# =============================================================================
# New Combined Models
# =============================================================================
class XGBRidgeCVRegressor(BaseEstimator, RegressorMixin):
    """
    Combined model that uses XGBRegressor for feature selection
    (by computing feature importances) and then fits a RidgeCV
    on the selected features.
    """
    def __init__(self, importance_threshold=None, **xgb_params):
        """
        Parameters:
            importance_threshold: float or None (default)
                The threshold to select features. If None, the median
                importance is used.
            xgb_params: dict
                Additional keyword parameters for XGBRegressor.
        """
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
        self.ridge = RidgeCV()
        self.ridge.fit(X[:, self.feature_mask_], y)
        return self

    def predict(self, X):
        if self.feature_mask_ is None:
            raise ValueError("The model has not been fitted yet.")
        return self.ridge.predict(X[:, self.feature_mask_])


class RFQuantileRegressor(BaseEstimator, RegressorMixin):
    """
    Custom estimator that wraps a RandomForestRegressor to perform
    quantile regression by computing the desired quantile over the
    predictions of all trees.
    """
    def __init__(self, quantile=0.5, **rf_kwargs):
        """
        Parameters:
            quantile: float (default 0.5)
                The quantile to predict (0.0 to 1.0); 0.5 corresponds to the median.
            rf_kwargs: dict
                Additional keyword parameters for RandomForestRegressor.
        """
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
# Data loading and preprocessing for new input/target datasets
# =============================================================================
def load_and_preprocess_data(target_path, input_path):
    """
    Loads the target (NOAA ERSV5) and input (NOAA OISST) datasets,
    aligns their time coordinates by shifting both to the month start,
    selects the common time range, cleans missing values, loads the data 
    into memory, and plots the SST maps for the last available month.
    
    It also prints the min and max SST values (ignoring NaNs) before and 
    after cleaning.
    
    Additionally, for debugging purposes:
      - Prints the values at lat = 0, lon = 0 for the before last month.
      - Checks and prints if the (0,0) grid cell contains NaN in any months.
    
    Parameters:
        target_path (str): Path to the target dataset.
        input_path (str): Path to the input dataset.
        
    Returns:
        target_data (xarray.DataArray): Cleaned SST data from the target dataset.
        input_data (xarray.DataArray): Cleaned SST data from the input dataset.
    """
    import xarray as xr
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from dask.diagnostics import ProgressBar

    # Open the datasets with xarray; decode times and use dask chunks along time.
    target_ds = xr.open_dataset(target_path, decode_times=True, chunks={"time": 1})
    input_ds = xr.open_dataset(input_path, decode_times=True, chunks={"time": 1})
    
    def shift_time_to_month_start(time_dataarray):
        times = pd.to_datetime(time_dataarray.values)
        return times.to_period("M").to_timestamp()  # Shift time to month start.
    
    # Shift the time coordinates for both datasets.
    target_ds = target_ds.assign_coords(time=("time", shift_time_to_month_start(target_ds["time"])))
    input_ds = input_ds.assign_coords(time=("time", shift_time_to_month_start(input_ds["time"])))
    
    # Select the common time range.
    tmin = max(target_ds.time.values[0], input_ds.time.values[0])
    tmax = min(target_ds.time.values[-1], input_ds.time.values[-1])
    target_ds = target_ds.sel(time=slice(tmin, tmax))
    input_ds = input_ds.sel(time=slice(tmin, tmax))
    common_times = np.intersect1d(target_ds.time.values, input_ds.time.values)
    if len(common_times) == 0:
        raise ValueError("No overlapping times found after aligning datasets.")
    target_ds = target_ds.sel(time=common_times)
    input_ds = input_ds.sel(time=common_times)
    
    # ----- Process Target SST -----
    raw_target = target_ds["sst"].values
    print("Raw target sst: min = {}, max = {}".format(np.nanmin(raw_target), np.nanmax(raw_target)))
    
    # Clean missing values: replace non-finite values and known missing-value flag.
    NOAA_MISSING = -9.96921e+36
    target_sst = target_ds["sst"].where(np.isfinite(target_ds["sst"]), np.nan)
    target_sst = target_sst.where(target_sst != NOAA_MISSING, np.nan)
    clean_target = target_sst.values
    print("Cleaned target sst: min = {}, max = {}".format(np.nanmin(clean_target), np.nanmax(clean_target)))
    
    # ----- Process Input (OISST) SST -----
    raw_input = input_ds["sst"].values
    print("Raw input sst: min = {}, max = {}".format(np.nanmin(raw_input), np.nanmax(raw_input)))
    
    input_sst = input_ds["sst"].where(np.isfinite(input_ds["sst"]), np.nan)
    input_sst = input_sst.where(input_sst != -999, np.nan)
    clean_input = input_sst.values
    print("Cleaned input sst: min = {}, max = {}".format(np.nanmin(clean_input), np.nanmax(clean_input)))
    
    # Load the cleaned data into memory.
    with ProgressBar():
        target_data = target_sst.load()
        input_data = input_sst.load()
    
    # -------------------- DEBUGGING ADDED CODE --------------------
    # For debugging, use the before last month (index -2) as the target time.
    before_last_time = target_data["time"].values[-2]
    
    # Select the (0,0) grid cell for that month using nearest-neighbor selection.
    # (Assumes that the coordinates are named "lat" and "lon".)
    target_val_00 = target_data.sel(time=before_last_time, lat=0, lon=0, method="nearest")
    input_val_00 = input_data.sel(time=before_last_time, lat=0, lon=0, method="nearest")
    print(f"DEBUG: Target (0,0) value for before last month ({before_last_time}): {target_val_00.values}")
    print(f"DEBUG: Input (0,0) value for before last month ({before_last_time}): {input_val_00.values}")
    
    # Check if the (0,0) grid cell contains NaN in any month for the target data.
    target_00_all = target_data.sel(lat=0, lon=0, method="nearest")
    nan_mask_target = np.isnan(target_00_all.values)
    if np.any(nan_mask_target):
        print("DEBUG: Target grid cell (0,0) is NaN in the following months:")
        for t in target_00_all.time.values[nan_mask_target]:
            print(pd.to_datetime(t).strftime("%Y-%m"))
    else:
        print("DEBUG: Target grid cell (0,0) is never NaN in any month.")
    
    # Check if the (0,0) grid cell contains NaN in any month for the input data.
    input_00_all = input_data.sel(lat=0, lon=0, method="nearest")
    nan_mask_input = np.isnan(input_00_all.values)
    if np.any(nan_mask_input):
        print("DEBUG: Input grid cell (0,0) is NaN in the following months:")
        for t in input_00_all.time.values[nan_mask_input]:
            print(pd.to_datetime(t).strftime("%Y-%m"))
    else:
        print("DEBUG: Input grid cell (0,0) is never NaN in any month.")
    # ------------------ END DEBUGGING CODE ----------------------
    
    # ----- Plot the Last Month for Each Dataset -----
    # (Using the same before_last_time as in the debugging output.)
    last_time = before_last_time
    
    # Select the month's data (should be 2D: lat x lon).
    target_last = target_data.sel(time=last_time)
    input_last = input_data.sel(time=last_time)
    
    # Format the time for the plot titles.
    time_str = pd.to_datetime(last_time).strftime("%Y-%m")
    
    # Create subplots to display the target and input SST maps side by side.
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Target SST.
    im1 = target_last.plot(ax=axs[0], cmap='viridis', add_colorbar=False)
    axs[0].set_title(f"Target SST for {time_str}")
    cbar1 = fig.colorbar(im1, ax=axs[0], orientation='vertical')
    cbar1.set_label("SST")
    
    # Plot Input SST.
    im2 = input_last.plot(ax=axs[1], cmap='viridis', add_colorbar=False)
    axs[1].set_title(f"Input SST for {time_str}")
    cbar2 = fig.colorbar(im2, ax=axs[1], orientation='vertical')
    cbar2.set_label("SST")
    
    plt.tight_layout()
    plt.show()
    
    return target_data, input_data


# =============================================================================
# GeoProcessor
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
        lat_diff = np.abs(self.lats - target_lat)
        lat_idx_unsorted = np.argpartition(lat_diff, n - 1)[:n]
        lat_idx = np.sort(lat_idx_unsorted)

        target_lon = self.wrap_longitude(target_lon)
        lon_diff = np.abs(self.lons - target_lon)
        lon_diff = np.minimum(lon_diff, 360 - lon_diff)
        lon_idx_unsorted = np.argpartition(lon_diff, n - 1)[:n]
        lon_idx = np.sort(lon_idx_unsorted)

        return lat_idx, lon_idx

# =============================================================================
# Configuration and training utilities
# =============================================================================
def generate_valid_patch_sizes():
    sizes = set()
    for a in range(7):  # 2^6 = 64
        for b in range(5):  # 3^4 = 81
            for c in range(3):  # 5^2 = 25
                size = (2 ** a) * (3 ** b) * (5 ** c)
                if 1 <= size <= 32:
                    sizes.add(size)
    return sorted(sizes)

# Dictionary mapping model names to classes.
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.dummy import DummyRegressor

def get_stacking():
    base_estimators = [("rf", RandomForestRegressor())]
    return StackingRegressor(estimators=base_estimators, final_estimator=Ridge())

# Here we update the mapping to include our new combined models.
types_mapping = {
    # "ridgecv": RidgeCV,
    # "mlp": MLPRegressor,
    # "rf_quantile": RFQuantileRegressor,
    # "ridge": Ridge,
    # "lasso": Lasso,
    # "elasticnet": ElasticNet,
    # "elasticnetcv": ElasticNetCV,  # Added ElasticNetCV
    # "xgboost": XGBRegressor,
    # "random_forest": RandomForestRegressor,
    # "dummy": DummyRegressor,
    # "lgbm": LGBMRegressor,
    # "catboost": CatBoostRegressor,
    # "gradient_boosting": GradientBoostingRegressor,
    # "extra_trees": ExtraTreesRegressor,
    # "knn": KNeighborsRegressor,
    # "svr": SVR,
    # "gpr": GaussianProcessRegressor,
    # "linear": LinearRegression,
    # "huber": HuberRegressor,
    # "bayesianridge": BayesianRidge,
    # "theilsen": TheilSenRegressor,
    # "ransac": RANSACRegressor,
    # "quantile": QuantileRegressor,
    # "lars": Lars,
    # "lassolars": LassoLars,
    # "omp": OrthogonalMatchingPursuit,
    "omp_cv": OrthogonalMatchingPursuitCV,
    # "ard": ARDRegression,
    # "pls": PLSRegression,        
 
}

MODEL_TYPES = list(types_mapping.keys())
SCALER_TYPES = [None, "standard", "minmax"]
PATCH_SIZES = [32]

def get_random_config():
    return (
        random.choice(MODEL_TYPES),
        random.choice(SCALER_TYPES),
        random.choice(PATCH_SIZES)
    )

# =============================================================================
# Timeout-protected training (for the model training step)
# =============================================================================
def train_worker(model_type, X, y, scaler_type, queue):
    try:
        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}")

        model = types_mapping[model_type]()
        if scaler_type == "standard":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        else:
            scaler = None

        model.fit(X, y)
        queue.put((model, scaler, None))
    except Exception as e:
        queue.put((None, None, e))

def train_with_timeout(model_type, X, y, scaler_type, timeout=150):
    q = Queue()
    p = Process(target=train_worker, args=(model_type, X, y, scaler_type, q))
    p.start()

    try:
        result = q.get(timeout=timeout)
    except Exception as e:
        if p.is_alive():
            p.kill()
            p.join()
        return None, None, "Timeout"

    if p.is_alive():
        p.join()

    if result[2] is not None:
        print("Caught exception during model training:", result[2])
        return None, None, str(result[2])
    
    return result[0], result[1], None

# =============================================================================
# Model Manager
# =============================================================================
from tqdm import tqdm  # make sure tqdm is imported

class ModelManager:
    def __init__(self, nlat, nlon):
        self.nlat = nlat
        self.nlon = nlon
        # Initialize grids for RMSE and R2.
        self.rmse_grid = np.full((nlat, nlon), np.inf)
        self.r2_grid = np.full((nlat, nlon), -np.inf)
        self._load_existing_models()

    def _load_existing_models(self):
        """
        Loads all existing model files from MODEL_DIR.
        For each model file, the filename is assumed to follow the pattern:
            "{lat}_{lon}_{metric_type}.joblib"
        where metric_type is either 'rmse' or 'r2'. The method then updates the
        rmse_grid and r2_grid accordingly. A tqdm progress bar is displayed.
        """
        curr = None
        rr = -Infinity
        # Get all model files.
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")]
        for f in tqdm(model_files, desc="Loading existing models"):
            try:
                # Expecting filenames like "i_j_metric.joblib"
                parts = f[:-7].split("_")
                if len(parts) < 3:
                    print("Skipping file with unexpected format:", f)
                    continue
                lat, lon, metric_type = parts[0], parts[1], parts[2]
                i, j = int(lat), int(lon)
                data = joblib.load(os.path.join(MODEL_DIR, f))
                if "feature_mask" not in data or data["feature_mask"] is None:
                    print(f"Warning: Model file {f} has no feature_mask; please update it using update_missing_feature_masks().")
                # Update grids based on the metric.
                if metric_type == "rmse" and data["rmse"] < self.rmse_grid[i, j]:
                    self.rmse_grid[i, j] = data["rmse"]
                    if data["rmse"] > rr:
                        rr = data["rmse"]
                        curr = data["model_type"]
                elif metric_type == "r2" and data["r2"] > self.r2_grid[i, j]:
                    self.r2_grid[i, j] = data["r2"]
            except AttributeError as e:
                # If the file is corrupted (for example, a known pandas issue), delete it.
                if "pandas.core.strings" in str(e) and "StringMethods" in str(e):
                    print(f"Corrupted file detected: {f}, deleting...")
                    os.remove(os.path.join(MODEL_DIR, f))
                else:
                    print(f"Error loading {f}: {str(e)}")
            except Exception as e:
                print(f"Error loading {f}: {str(e)}")
        print("Best model type (for the highest rmse):", curr, "with RMSE:", rr)

    def save_model(self, i, j, model_type, scaler_type, patch_size, model, scaler, rmse, r2, feature_mask):
        """
        Saves the model if it improves on the existing metrics.
        The model_dict is saved using a filename that encodes the pixel (i,j)
        and the metric (either rmse or r2).
        """
        improved = False
        print("Current metrics:", rmse, r2, "Existing:", self.rmse_grid[i, j], self.r2_grid[i, j])
        model_dict = {
            "model": model,
            "scaler": scaler,
            "model_type": model_type,
            "scaler_type": scaler_type,
            "patch_size": patch_size,
            "rmse": rmse,
            "r2": r2,
            "feature_mask": feature_mask
        }
        if rmse < self.rmse_grid[i, j]:
            self.rmse_grid[i, j] = rmse
            joblib.dump(model_dict, os.path.join(MODEL_DIR, f"{i}_{j}_rmse.joblib"))
            improved = True
        if r2 > self.r2_grid[i, j]:
            self.r2_grid[i, j] = r2
            joblib.dump(model_dict, os.path.join(MODEL_DIR, f"{i}_{j}_r2.joblib"))
            improved = True
        return improved

# =============================================================================
# Resumable Trainer
# =============================================================================
import os
import random
import threading
import concurrent.futures
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Assume that MODEL_DIR, RANDOM_SEED, load_and_preprocess_data, ModelManager,
# GeoProcessor, get_random_config, and train_with_timeout are defined elsewhere.

class ResumableTrainer:
    def __init__(self):

        # Here, the target dataset is assumed to be NOAA ERSV5 (e.g., "sst.mnmean.nc")
        # and the input dataset is the merged NOAA OISST (e.g., "oisst_monthly.nc").
        target_data, input_data = load_and_preprocess_data("sst.mnmean.nc", "oisst_monthly.nc")
        nlat, nlon = len(target_data.lat), len(target_data.lon)
        print("start model manager")
        self.model_mgr = ModelManager(nlat, nlon)
        # Use target data coordinates for selecting the target pixel.
        self.geo = GeoProcessor(target_data.lat.values, target_data.lon.values)
        # IMPORTANT: Create a separate geo processor for the input dataset!
        self.input_geo = GeoProcessor(input_data.lat.values, input_data.lon.values)

        # Train/test split: Exclude December 2024 from training.
        date_mask_dec = (target_data.time.dt.year == 2024) & (target_data.time.dt.month == 12)
        train_val_mask = ~date_mask_dec
        target_train_val = target_data.sel(time=train_val_mask)
        input_train_val = input_data.sel(time=train_val_mask)
        train_times, test_times = train_test_split(
            target_train_val.time.values, test_size=0.2, random_state=RANDOM_SEED
        )
        self.target_train = target_train_val.sel(time=train_times)
        self.input_train = input_train_val.sel(time=train_times)
        self.target_test = target_train_val.sel(time=test_times)
        self.input_test = input_train_val.sel(time=test_times)

        # Path for saving the hopeless set (stored in MODEL_DIR)
        self.hopeless_set_path = "hopeless_set.joblib"

        # Load the hopeless pixels set from disk if it exists; otherwise, create a new set.
        if os.path.exists(self.hopeless_set_path):
            try:
                self.hopeless_pixels = joblib.load(self.hopeless_set_path)
                print(f"Loaded hopeless set with {len(self.hopeless_pixels)} pixels.")
            except Exception as e:
                print("Failed to load hopeless set, starting with an empty set. Error:", e)
                self.hopeless_pixels = set()
        else:
            self.hopeless_pixels = set()

        # Track pixels in progress.
        self.in_progress = set()
        self.lock = threading.Lock()

        # Update any previously saved models that lack a feature mask. [don't delete this, uncomment if needed]
        # self.update_missing_feature_masks()

    def save_hopeless_set(self):
        """Persist the hopeless pixels set to disk."""
        try:
            joblib.dump(self.hopeless_pixels, self.hopeless_set_path)
            # Optionally, you can print or log that the set was saved:
            # print(f"Saved hopeless set with {len(self.hopeless_pixels)} pixels.")
        except Exception as e:
            print("Error saving hopeless set:", e)

    def compute_feature_mask_for_pixel(self, i, j, patch_size):
        # Use the input dataset's geo processor to select the patch.
        lat_idx, lon_idx = self.input_geo.get_nearest_indices(
            self.target_train.lat[i].item(),
            self.target_train.lon[j].item(),
            patch_size
        )
        input_train_subset = self.input_train.isel(lat=lat_idx, lon=lon_idx)
        input_test_subset = self.input_test.isel(lat=lat_idx, lon=lon_idx)
        X_train = input_train_subset.values
        X_test = input_test_subset.values
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        cols_keep = np.all(~np.isnan(X_train_flat), axis=0) & np.all(~np.isnan(X_test_flat), axis=0)
        return cols_keep

    def update_missing_feature_masks(self):
        for f in os.listdir(MODEL_DIR):
            if f.endswith(".joblib"):
                file_path = os.path.join(MODEL_DIR, f)
                data = joblib.load(file_path)
                if "feature_mask" not in data or data["feature_mask"] is None:
                    try:
                        parts = f[:-7].split("_")
                        if len(parts) < 3:
                            print("Skipping file with unexpected format:", f)
                            continue
                        i = int(parts[0])
                        j = int(parts[1])
                        patch_size = data.get("patch_size", None)
                        if patch_size is None:
                            print("No patch_size found in model", f, "skipping mask update")
                            continue
                        feature_mask = self.compute_feature_mask_for_pixel(i, j, patch_size)
                        data["feature_mask"] = feature_mask
                        joblib.dump(data, file_path)
                        print("Updated feature mask for model", f)
                    except Exception as e:
                        print("Error updating mask for model", f, ":", e)

    def get_worst_pixel(self, exclude_set):
        # Create a boolean mask of valid pixels.
        mask = np.ones(self.model_mgr.rmse_grid.shape, dtype=bool)
        for (i, j) in exclude_set:
            mask[i, j] = False

        if not np.any(mask):
            return None

        # With 50% probability, simply choose a random valid pixel.
        if random.random() < 1:
            valid_indices = np.argwhere(mask)
            return tuple(random.choice(valid_indices))

        # --- For RMSE: higher values are worse ---
        # Replace invalid entries with -infinity.
        rmse_valid = np.where(mask, self.model_mgr.rmse_grid, -np.inf)
        rmse_flat = rmse_valid.flatten()
        valid_rmse = np.where(rmse_flat != -np.inf)[0]
        rmse_k = 200
        if len(valid_rmse) == 0:
            top3_rmse_pixels = []
        else:
            # If there are fewer than 3 valid pixels, just use them all.
            if len(valid_rmse) <= rmse_k:
                top3_rmse_indices = valid_rmse
            else:
                # Get the indices of the top 3 worst (largest) RMSE values.
                top3_rmse_indices = np.argpartition(rmse_flat, -rmse_k)[-rmse_k:]
            # Convert flat indices back to 2D indices.
            top3_rmse_pixels = [np.unravel_index(idx, self.model_mgr.rmse_grid.shape)
                                for idx in top3_rmse_indices]

        # --- For R²: lower values are worse ---
        # Replace invalid entries with +infinity.
        r2_valid = np.where(mask, self.model_mgr.r2_grid, np.inf)
        r2_flat = r2_valid.flatten()
        valid_r2 = np.where(r2_flat != np.inf)[0]
        r2_k = 200
        if len(valid_r2) == 0:
            top3_r2_pixels = []
        else:
            if len(valid_r2) <= r2_k:
                top3_r2_indices = valid_r2
            else:
                # Get the indices of the top 3 worst (smallest) R² values.
                top3_r2_indices = np.argpartition(r2_flat, r2_k)[:r2_k]
            top3_r2_pixels = [np.unravel_index(idx, self.model_mgr.r2_grid.shape)
                            for idx in top3_r2_indices]

        # Combine candidates from both metrics.
        candidates = top3_rmse_pixels + top3_r2_pixels
        if not candidates:
            return None

        return random.choice(candidates)


    def train_pixel(self, i, j):
        try:
            model_type, scaler_type, patch_size = get_random_config()
            print(f"Training pixel ({i}, {j}) with model: {model_type}, scaler: {scaler_type}, patch_size: {patch_size}")

            # Select the target pixel using its coordinate values.
            lat_val = self.target_train.lat.values[i]
            lon_val = self.target_train.lon.values[j]
            y_train = self.target_train.sel(lat=lat_val, lon=lon_val, method="nearest").values
            y_test = self.target_test.sel(lat=lat_val, lon=lon_val, method="nearest").values

            if np.isnan(y_train).all() or np.isnan(y_test).all():
                print(f"Pixel ({i},{j}) hopeless: All y_train or y_test values are NaN.")
                with self.lock:
                    self.hopeless_pixels.add((i, j))
                    self.save_hopeless_set()
                return False

            # Use the input dataset's geo processor to extract a patch.
            lat_idx, lon_idx = self.input_geo.get_nearest_indices(lat_val, lon_val, patch_size)

            # Extract input (OISST) features.
            input_train_subset = self.input_train.isel(lat=lat_idx, lon=lon_idx)
            input_test_subset = self.input_test.isel(lat=lat_idx, lon=lon_idx)
            X_train = input_train_subset.values
            X_test = input_test_subset.values

            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)

            cols_keep = np.all(~np.isnan(X_train_flat), axis=0) & np.all(~np.isnan(X_test_flat), axis=0)
            if not np.any(cols_keep):
                print(f"Pixel ({i},{j}) hopeless: All features contain NaNs.")
                # Optionally, add to hopeless set here as well:
                # with self.lock:
                #     self.hopeless_pixels.add((i, j))
                #     self.save_hopeless_set()
                return False

            X_train_clean = X_train_flat[:, cols_keep]
            X_test_clean = X_test_flat[:, cols_keep]

            valid_train = ~np.isnan(y_train)
            if valid_train.sum() < 1:
                print(f"Pixel ({i},{j}) hopeless: Not enough valid training samples.")
                with self.lock:
                    self.hopeless_pixels.add((i, j))
                    self.save_hopeless_set()
                return False

            X_train_valid = X_train_clean[valid_train, :]
            y_train_valid = y_train[valid_train]

            model, scaler, error = train_with_timeout(model_type, X_train_valid, y_train_valid, scaler_type)
            if error or model is None:
                print(f"Error during training pixel ({i},{j}):", error)
                return False

            valid_test = ~np.isnan(y_test)
            if valid_test.sum() < 1:
                print(f"Pixel ({i},{j}): No valid testing samples.")
                return False

            X_test_valid = X_test_clean[valid_test, :]
            y_test_valid = y_test[valid_test]

            if scaler:
                X_test_valid = scaler.transform(X_test_valid)

            y_pred_test = model.predict(X_test_valid)
            rmse = np.sqrt(mean_squared_error(y_test_valid, y_pred_test))
            r2 = r2_score(y_test_valid, y_pred_test)

            print(f"Evaluation Metrics for pixel ({i},{j}) - RMSE: {rmse}, R2: {r2}")

            improved = self.model_mgr.save_model(
                i, j, model_type, scaler_type, patch_size, model, scaler, rmse, r2,
                feature_mask=cols_keep
            )
            return improved

        except Exception as e:
            print(f"An unexpected error occurred for pixel ({i},{j}): {e}")
            return False

    def run(self, max_iterations=100000, n_workers=4):
        futures_dict = {}
        iterations = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            while iterations < max_iterations:
                with self.lock:
                    exclude = self.hopeless_pixels.union(self.in_progress)
                    worst_pixel = self.get_worst_pixel(exclude)
                    if worst_pixel is None:
                        print("No more improvable pixels available.")
                        break
                    i, j = worst_pixel
                    self.in_progress.add((i, j))
                future = executor.submit(self.train_pixel, i, j)
                futures_dict[future] = (i, j)
                iterations += 1

                done_futures = []
                for fut in concurrent.futures.as_completed(list(futures_dict.keys())):
                    pix = futures_dict[fut]
                    try:
                        _ = fut.result()
                    except Exception as e:
                        print(f"Pixel {pix} training raised exception: {e}")
                    with self.lock:
                        self.in_progress.discard(pix)
                    done_futures.append(fut)
                    if len(done_futures) > 0:
                        break
                for fut in done_futures:
                    futures_dict.pop(fut, None)

            for fut in concurrent.futures.as_completed(futures_dict.keys()):
                pix = futures_dict[fut]
                try:
                    _ = fut.result()
                except Exception as e:
                    print(f"Pixel {pix} training exception during final wait: {e}")
                with self.lock:
                    self.in_progress.discard(pix)
                futures_dict.pop(fut, None)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    trainer = ResumableTrainer()
    trainer.run(max_iterations=100000, n_workers=4)
