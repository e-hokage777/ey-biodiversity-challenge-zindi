import xarray as xr
import planetary_computer as pc
import pystac_client
from argparse import ArgumentParser
import os


def sanitize_attrs(attrs):
    clean = {}
    for k, v in attrs.items():
        if isinstance(v, str):
            clean[k] = v.encode("utf-8", errors="ignore").decode("utf-8")
        else:
            clean[k] = v
    return clean


def download_terraclimate_dataset(
    output_path, time_start: str | None = None, time_end: str | None = None
):
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    collection = catalog.get_collection("terraclimate")
    asset = collection.assets["zarr-abfs"]

    open_kwargs = asset.extra_fields["xarray:open_kwargs"]
    open_kwargs["storage_options"] = {
        **open_kwargs.get("storage_options", {}),
        "connection_timeout": 600,
        "read_timeout": 600,
        "max_retries": 10,
    }

    min_lon, max_lon = 139.94, 151.48
    min_lat, max_lat = -39.74, -30.92

    ## setting up time slice
    time_slice = slice(time_start, time_end)

    # First open just to get variable names
    print("Fetching dataset metadata...")
    with xr.open_dataset(asset.href, **open_kwargs) as ds:
        variables = list(ds.data_vars)
    print(f"Found {len(variables)} variables: {variables}")

    datasets = []
    for i, var in enumerate(variables):
        print(f"[{i+1}/{len(variables)}] Downloading variable: {var}")
        try:
            with xr.open_dataset(asset.href, **open_kwargs) as ds:
                ds = ds[[var]].drop_vars("crs", errors="ignore")
                ds = ds.sel(time=time_slice)

                mask_lon = (ds.lon >= min_lon) & (ds.lon <= max_lon)
                mask_lat = (ds.lat >= min_lat) & (ds.lat <= max_lat)
                ds = ds.where(mask_lon & mask_lat, drop=True)

                # Sanitize attributes
                ds.attrs = sanitize_attrs(ds.attrs)
                ds[var].attrs = sanitize_attrs(ds[var].attrs)
                for coord in ds.coords:
                    ds[coord].attrs = sanitize_attrs(ds[coord].attrs)

                # Force into memory before closing the remote connection
                ds = ds.load()
                datasets.append(ds)
                print(f"  ✓ {var} loaded successfully")

        except Exception as e:
            print(f"  ✗ Failed to download {var}: {e}. Skipping.")
            continue

    if not datasets:
        raise RuntimeError("No variables were successfully downloaded.")

    print(f"\nMerging {len(datasets)} variables...")
    combined = xr.merge(datasets)

    # Sanitize global attrs on the merged dataset
    combined.attrs = sanitize_attrs(combined.attrs)

    print(f"Saving to {output_path}...")
    combined.to_netcdf(output_path)
    print("Done.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output-path", type=str, required=True, help="Output path for NetCDF file"
    )
    parser.add_argument(
        "--time-start", type=str, default=None, help="Start time for time slice"
    )
    parser.add_argument(
        "--time-end", type=str, default=None, help="End time for time slice"
    )
    parser.add_argument(
        "--sample-time-range", action="store_true", help="Download all variables"
    )

    args = parser.parse_args()

    dir_name = os.path.dirname(args.output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if args.sample_time_range:
        download_terraclimate_dataset(
            output_path=args.output_path,
            time_start="2017-11-01",
            time_end="2019-11-01",
        )
    else:
        download_terraclimate_dataset(
            output_path=args.output_path,
            time_start=args.time_start,
            time_end=args.time_end,
        )
