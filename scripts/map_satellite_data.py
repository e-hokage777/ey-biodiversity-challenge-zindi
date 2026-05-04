# Extracts satellite variables from a GeoTIFF based on coordinates from a csv file and returns them in a DataFrame.

import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from argparse import ArgumentParser
from averaging_out import reduce_array


def map_satellite_data(ds_path: str, csv_path: str, test: bool = False) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Open the GeoTIFF file and load data into xarray DataArrays
    ds = xr.open_dataset(ds_path, engine="netcdf4")

    # ds = ds.median(
    #     dim="time"
    # )  # TODO: Make sure to find better typical values by looking at values over a longer period of time
    ds = reduce_array(ds)

    rows = []

    # Extract values for each row in the DataFrame
    for _, row in tqdm(
        df.iterrows(), total=df.shape[0], desc="Extracting raster values"
    ):
        target_lat, target_lon = row["Latitude"], row["Longitude"]

        row_entries = {"latitude": target_lat, "longitude": target_lon}

        if not test:
            row_entries["Occurrence Status"] = row["Occurrence Status"]

        vars = list(ds.data_vars)

        for var in vars:
            try:
                row_entries[var] = (
                    ds[var]
                    .sel(lat=target_lat, lon=target_lon, method="nearest")
                    .values.item()
                )
            except:
                row_entries[var] = np.nan

        rows.append(row_entries)

    # Create a DataFrame to store the variable values
    df_new = pd.DataFrame(rows)

    return df_new


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output-path", type=str, required=True, help="Output path for NetCDF file"
    )
    parser.add_argument("--csv-path", type=str, required=True, help="Path to csv file")
    parser.add_argument(
        "--dataset-path", type=str, required=True, help="Path to GeoTIFF file"
    )
    parser.add_argument("--test", action="store_true", help="Test mode")

    args = parser.parse_args()

    df = map_satellite_data(args.dataset_path, args.csv_path, args.test)

    df.to_csv(args.output_path, index=False)
